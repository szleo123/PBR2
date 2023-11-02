#version 330 core

uniform float u_Time;

uniform vec3 u_CamPos;
uniform vec3 u_Forward, u_Right, u_Up;
uniform vec2 u_ScreenDims;

// PBR material attributes
uniform vec3 u_Albedo;
uniform float u_Metallic;
uniform float u_Roughness;
uniform float u_AmbientOcclusion;
// Texture maps for controlling some of the attribs above, plus normal mapping
uniform sampler2D u_AlbedoMap;
uniform sampler2D u_MetallicMap;
uniform sampler2D u_RoughnessMap;
uniform sampler2D u_AOMap;
uniform sampler2D u_NormalMap;
// If true, use the textures listed above instead of the GUI slider values
uniform bool u_UseAlbedoMap;
uniform bool u_UseMetallicMap;
uniform bool u_UseRoughnessMap;
uniform bool u_UseAOMap;
uniform bool u_UseNormalMap;

// Image-based lighting
uniform samplerCube u_DiffuseIrradianceMap;
uniform samplerCube u_GlossyIrradianceMap;
uniform sampler2D u_BRDFLookupTexture;

// Varyings
in vec2 fs_UV;
out vec4 out_Col;

const float PI = 3.14159f;

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct BSDF {
    vec3 pos;
    vec3 nor;
    vec3 albedo;
    float metallic;
    float roughness;
    float ao;
    float thinness;
};

struct MarchResult {
    float t;
    int hitSomething;
    BSDF bsdf;
};

struct SmoothMinResult {
    float dist;
    float material_t;
};

float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

float sceneSDF(vec3 query, bool n);

vec3 SDF_Normal(vec3 query) {
    vec2 epsilon = vec2(0.0, 0.001);
    return normalize( vec3( sceneSDF(query + epsilon.yxx, true) - sceneSDF(query - epsilon.yxx, true),
                            sceneSDF(query + epsilon.xyx, true) - sceneSDF(query - epsilon.xyx, true),
                            sceneSDF(query + epsilon.xxy, true) - sceneSDF(query - epsilon.xxy, true)));
}

float SDF_Sphere(vec3 query, vec3 center, float radius) {
    return length(query - center) - radius;
}

float SDF_Box(vec3 query, vec3 bounds ) {
  vec3 q = abs(query) - bounds;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float SDF_RoundCone( vec3 query, vec3 a, vec3 b, float r1, float r2) {
  // sampling independent computations (only depend on shape)
  vec3  ba = b - a;
  float l2 = dot(ba,ba);
  float rr = r1 - r2;
  float a2 = l2 - rr*rr;
  float il2 = 1.0/l2;

  // sampling dependant computations
  vec3 pa = query - a;
  float y = dot(pa,ba);
  float z = y - l2;
  float x2 = dot2( pa*l2 - ba*y );
  float y2 = y*y*l2;
  float z2 = z*z*l2;

  // single square root!
  float k = sign(rr)*rr*rr*x2;
  if( sign(z)*a2*z2>k ) return  sqrt(x2 + z2)        *il2 - r2;
  if( sign(y)*a2*y2<k ) return  sqrt(x2 + y2)        *il2 - r1;
                        return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

float SDF_Torus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float SDF_Cylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float smooth_min( float a, float b, float k ) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

SmoothMinResult smooth_min_lerp( float a, float b, float k ) {
    float h = max( k-abs(a-b), 0.0 )/k;
    float m = h*h*0.5;
    float s = m*k*0.5;
    if(a < b) {
        return SmoothMinResult(a-s,m);
    }
    return SmoothMinResult(b-s,1.0-m);
}
vec3 repeat(vec3 query, vec3 cell) {
    return mod(query + 0.5 * cell, cell) - 0.5 * cell;
}

float subtract(float d1, float d2) {
    return max(d1, -d2);
}

float opIntersection( float d1, float d2 ) {
    return max(d1,d2);
}

float opOnion(float sdf, float thickness ) {
    return abs(sdf)-thickness;
}

vec3 rotateX(vec3 p, float angle) {
    angle = angle * 3.14159 / 180.f;
    float c = cos(angle);
    float s = sin(angle);
    return vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

vec3 rotateZ(vec3 p, float angle) {
    angle = angle * 3.14159 / 180.f;
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

float SDF_Stache(vec3 query) {
    float left = SDF_Sphere(query / vec3(1,1,0.3), vec3(0.2, -0.435, 3.5), 0.1) * 0.1;
    left = min(left, SDF_Sphere(query / vec3(1,1,0.3), vec3(0.45, -0.355, 3.5), 0.1) * 0.1);
    left = min(left, SDF_Sphere(query / vec3(1,1,0.3), vec3(0.7, -0.235, 3.5), 0.09) * 0.1);
    left = subtract(left, SDF_Sphere(rotateZ(query, -15) / vec3(1.3,1,1), vec3(0.3, -0.1, 1.), 0.35));

    float right = SDF_Sphere(query / vec3(1,1,0.3), vec3(-0.2, -0.435, 3.5), 0.1) * 0.1;
    right = min(right, SDF_Sphere(query / vec3(1,1,0.3), vec3(-0.45, -0.355, 3.5), 0.1) * 0.1);
    right = min(right, SDF_Sphere(query / vec3(1,1,0.3), vec3(-0.7, -0.235, 3.5), 0.09) * 0.1);
    right = subtract(right, SDF_Sphere(rotateZ(query, 15) / vec3(1.3,1,1), vec3(-0.3, -0.1, 1.), 0.35));

    return min(left, right);
}

float SDF_Wahoo_Skin(vec3 query) {
    // head base
    float result = SDF_Sphere(query / vec3(1,1.2,1), vec3(0,0,0), 1.) * 1.1;
    // cheek L
    result = smooth_min(result, SDF_Sphere(query, vec3(0.5, -0.4, 0.5), 0.5), 0.3);
    // cheek R
    result = smooth_min(result, SDF_Sphere(query, vec3(-0.5, -0.4, 0.5), 0.5), 0.3);
    // chin
    result = smooth_min(result, SDF_Sphere(query, vec3(0.0, -0.85, 0.5), 0.35), 0.3);
    // nose
    result = smooth_min(result, SDF_Sphere(query / vec3(1.15,1,1), vec3(0, -0.2, 1.15), 0.35), 0.05);
    return result;
}

float SDF_Wahoo_Hat(vec3 query) {
    float result = SDF_Sphere(rotateX(query, 20) / vec3(1.1,0.5,1), vec3(0,1.65,0.4), 1.);
    result = smooth_min(result, SDF_Sphere((query - vec3(0,0.7,-0.95)) / vec3(2.5, 1.2, 1), vec3(0,0,0), 0.2), 0.3);
    result = smooth_min(result, SDF_Sphere(query / vec3(1.5,1,1), vec3(0, 1.3, 0.65), 0.5), 0.3);

    float brim = opOnion(SDF_Sphere(query / vec3(1.02, 1, 1), vec3(0, -0.15, 1.), 1.1), 0.02);

    brim = subtract(brim, SDF_Box(rotateX(query - vec3(0, -0.55, 0), 10), vec3(10, 1, 10)));

    result = min(result, brim);

    return result;
}


float SDF_Wahoo(vec3 query) {
    // Flesh-colored parts
    float result = SDF_Wahoo_Skin(query);
    // 'stache parts
    result = min(result, SDF_Stache(query));
    // hat
    result = min(result, SDF_Wahoo_Hat(query));

    return result;
}

BSDF BSDF_Wahoo(vec3 query) {
    // Head base
    BSDF result = BSDF(query, normalize(query), pow(vec3(239, 181, 148) / 255., vec3(2.2)),
                       0., 0.7, 1., 0.);

    result.nor = SDF_Normal(query);

    float skin = SDF_Wahoo_Skin(query);
    float stache = SDF_Stache(query);
    float hat = SDF_Wahoo_Hat(query);

    if(stache < skin && stache < hat) {
        result.albedo = pow(vec3(68,30,16) / 255., vec3(2.2));
    }
    if(hat < skin && hat < stache) {
        result.albedo = pow(vec3(186,45,41) / 255., vec3(2.2));
    }

    return result;
}
///Custom/////////////
float SDF_Cube(vec3 query) {
    float result = SDF_Box(query, vec3(1.f));
    result = opIntersection(result, SDF_Sphere(query, vec3(0.f), 1.3f));
    return result;
}

float SDF_INNER(vec3 query) {
    float result = SDF_Cylinder(query, 1.2f, 0.7f);
    result = min(result, SDF_Cylinder(rotateX(query, 90), 1.2f, 0.7f));
    result = min(result, SDF_Cylinder(rotateZ(rotateX(query, 90), 90), 1.2f, 0.7f));
    return result;
}

float SDF_Hollow(vec3 query) {
    return subtract(SDF_Cube(query), SDF_INNER(query));
}

float SDF_Ring1(vec3 query) {
    float result = SDF_Sphere(query, vec3(0.f), 0.5);
    result = smooth_min(result, SDF_Cylinder(query, 2.2f, 0.3f), 0.05);
    result = smooth_min(result, SDF_Torus(rotateX(query, 90), vec2(2.3f, 0.3)), 0.1);
    return result;
}

float SDF_Ring2(vec3 query) {
    float result = SDF_Sphere(query, vec3(0.f), 0.5);
    result = smooth_min(result, SDF_Cylinder(rotateZ(rotateX(query, 90), 90), 1.5f, 0.3f), 0.05);
    result = smooth_min(result, SDF_Torus(query, vec2(1.6f, 0.3)), 0.1);
    return result;
}

float SDF_Final(vec3 query) {
    return min(SDF_Ring2(query), min(SDF_Ring1(query), SDF_Hollow(query)));
}





///////////////////////////////////////////////
// Noise functions
vec2 random2( vec2 p ) {
    return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}

vec3 random3( vec3 p ) {
    return fract(sin(vec3(dot(p,vec3(127.1,311.7,290.1)),dot(p,vec3(269.5,183.3,810.4)), dot(p,vec3(111.3,402.9,567.2))))*43758.5453);
}

float surflet(vec3 p, vec3 gridPoint) {
    // Compute the distance between p and the grid point along each axis, and warp it with a
    // quintic function so we can smooth our cells
    vec3 t2 = abs(p - gridPoint);
    vec3 t = vec3(1.f) - 6.f * pow(t2, vec3(5.f)) + 15.f * pow(t2, vec3(4.f)) - 10.f * pow(t2, vec3(3.f));
    // Get the random vector for the grid point (assume we wrote a function random2
    // that returns a vec2 in the range [0, 1])
    vec3 gradient = random3(gridPoint) * 2. - vec3(1., 1., 1.);
    // Get the vector from the grid point to P
    vec3 diff = p - gridPoint;
    // Get the value of our height field by dotting grid->P with our gradient
    float height = dot(diff, gradient);
    // Scale our height field (i.e. reduce it) by our polynomial falloff function
    return height * t.x * t.y * t.z;
}

float perlinNoise3D(vec3 p) {
        float surfletSum = 0.f;
        // Iterate over the four integer corners surrounding uv
        for(int dx = 0; dx <= 1; ++dx) {
                for(int dy = 0; dy <= 1; ++dy) {
                        for(int dz = 0; dz <= 1; ++dz) {
                                surfletSum += surflet(p, floor(p) + vec3(dx, dy, dz));
                        }
                }
        }
        return surfletSum;
}

////////////////////////////////////////////////

float gridSDF(vec3 query) {
    float out_dist = 1e20;
    const float grid_scale = 1.f;
    query *= grid_scale;

    vec2 cellID = floor(query.xz);
    vec2 cellFract = fract(query.xz);
    for (int x = -1; x < 2; ++x) {
        for (int z = -1; z < 2; ++z) {
            vec2 neighborCell = vec2(float(x), float(z)) + cellID;

            vec2 randPt = random2(neighborCell);
            vec2 neighborPt = neighborCell + randPt;

//            float dist = SDF_Sphere(query - vec3(neighborPt.x, 0, neighborPt.y), vec3(0.f), 0.2f);
            float dist = SDF_Final(query - vec3(neighborPt.x, 0, neighborPt.y));
            if (out_dist == 1e20) {
                out_dist = dist;
            } else {
                out_dist = smooth_min(out_dist, dist, 0.);
            }
        }
    }
    out_dist /= grid_scale;
    return out_dist;
}

float sceneSDF(vec3 query, bool n) {
    // the following 2 lines for infinite repetition
    float result;
    result = SDF_Final(query);
    if (n) return result;
    vec3 c = vec3(10.f);
    vec3 cellID = floor((query + 0.5 * c) / c); // cell ID
    vec3 offset = (random3(cellID) * 2. - vec3(1.)) * 1.;
    query = mod(query+0.5*c,c)-0.5*c;
//    return SDF_Sphere(query, vec3(0.), 1.f);
//    return SDF_Wahoo(query);
    return SDF_Final(query - offset);
}


#define SINGLE 0

BSDF BSDF_Final(vec3 query, vec3 cellID) {
    // Head base
    vec3 offset = (random3(cellID) * 2. - vec3(1.)) * 1.;
#if SINGLE
    offset = vec3(0.);
#endif
    BSDF result = BSDF(query, SDF_Normal(query - offset), pow(vec3(170, 169, 173) / 255., vec3(2.2)),
                       1., 0.2, 1., 0.);
    float cube = SDF_Hollow(query - offset);
    float ring1 = SDF_Ring1(query - offset);
    float ring2 = SDF_Ring2(query - offset);

    if(ring1 < cube && ring1 < ring2) {
        result.albedo = pow(vec3(166,44,43) / 255., vec3(2.2));
#if SINGLE == 0
        if (mod(cellID.x + cellID.z, 2) == 0) {
            result.albedo = vec3(1.f);
        }
#endif
        result.metallic = 0.0;
    }
    if(ring2 < cube && ring2 < ring1) {
        result.albedo = pow(vec3(50,82,123) / 255., vec3(2.2));
#if SINGLE == 0
        if (mod(cellID.x + cellID.z, 2) == 0) {
            result.albedo = vec3(1.f);
        }
#endif
        result.metallic = 0.0;
    }
    if(ring1 == ring2) {
        result.albedo = pow(vec3(212,175,55) / 255., vec3(2.2));
    }


    return result;
}

BSDF sceneBSDF(vec3 query) {
    BSDF result;
    vec3 c = vec3(10.f);
    vec3 cellID = floor((query + 0.5 * c) / c); // add noise
    vec3 offset = (random3(cellID) * 2. - vec3(1.)) * 1.;
    query = mod(query+0.5*c,c)-0.5*c;
    result = BSDF(query, SDF_Normal(query), vec3(1, 1, 1),
                  0., 1.0, 1., 0.);
    result = BSDF_Final(query, cellID);
    return result;
}

// TODO add any helper functions you need here
void coordinateSystem(in vec3 v1, out vec3 v2, out vec3 v3) {
    if (abs(v1.x) > abs(v1.y))
            v2 = vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
        else
            v2 = vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
        v3 = cross(v1, v2);
}

mat3 LocalToWorld(vec3 nor) {
    vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return mat3(tan, bit, nor);
}

mat3 WorldToLocal(vec3 nor) {
    return transpose(LocalToWorld(nor));
}

void handleMaterialMaps(inout vec3 albedo, inout float metallic,
                        inout float roughness, inout float ambientOcclusion,
                        inout vec3 normal) {
    if(u_UseAlbedoMap) {
        albedo = pow(texture(u_AlbedoMap, fs_UV).rgb, vec3(2.2));
    }
    if(u_UseMetallicMap) {
        metallic = texture(u_MetallicMap, fs_UV).r;
    }
    if(u_UseRoughnessMap) {
        roughness = texture(u_RoughnessMap, fs_UV).r;
    }
    if(u_UseAOMap) {
        ambientOcclusion = texture(u_AOMap, fs_UV).r;
    }
    if(u_UseNormalMap) {
        // TODO: Apply normal mapping
       vec3 newNor = normalize(texture(u_NormalMap, fs_UV).rgb);
       normal = LocalToWorld(normal) * newNor;
    }
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}





vec3 metallic_plastic_LTE(BSDF bsdf, vec3 wo) {
    vec3 N = bsdf.nor;
    vec3 albedo = bsdf.albedo;
    float metallic = bsdf.metallic;
    float roughness = bsdf.roughness;
    float ambientOcclusion = bsdf.ao;

    // TODO
    handleMaterialMaps(albedo, metallic, roughness, ambientOcclusion, N);
    vec3 wi = reflect(-wo, N);

    // diffuse part
    vec3 R = albedo;
    vec3 diffuseIrradiance = texture(u_DiffuseIrradianceMap, N).rgb;

    // specular part
    vec3 plastic_CT_color = vec3(0.04);
    vec3 metallic_CT_color = albedo;
    vec3 fresnel_color = mix(plastic_CT_color, metallic_CT_color, metallic);
    vec3 F = fresnelSchlickRoughness(max(dot(N, wo), 0.0), fresnel_color, roughness);
    vec3 kS = F;
    vec3 kD = vec3(1.0f) - kS;
    kD *= 1.0 - metallic;
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_GlossyIrradianceMap, wi,  roughness * MAX_REFLECTION_LOD).rgb;
    vec2 envBRDF  = texture(u_BRDFLookupTexture, vec2(max(dot(N, wo), 0.0), roughness)).rg;
    vec3 specularIrradiance = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    vec3 result = kD * R * diffuseIrradiance + specularIrradiance;
    result *= ambientOcclusion;
//    result = result / (vec3(1.f) + result);
//    result = pow(result, vec3(1.f / 2.2f));
    return result;
}


#define FOVY 45 * PI / 180.f
Ray rayCast() {
    vec2 ndc = fs_UV;
    ndc = ndc * 2.f - vec2(1.f);

    float aspect = u_ScreenDims.x / u_ScreenDims.y;
    vec3 ref = u_CamPos + u_Forward;
    vec3 V = u_Up * tan(FOVY * 0.5);
    vec3 H = u_Right * tan(FOVY * 0.5) * aspect;
    vec3 p = ref + H * ndc.x + V * ndc.y;

    return Ray(u_CamPos, normalize(p - u_CamPos));
}

#define MAX_ITERATIONS 128
#define EPSILON 0.001
#define SCOPE 200
MarchResult raymarch(Ray ray) {
    float dist;
    float t = 0.001;
    int hitSomething = 0;

    float iter;
    for (iter = 0.0; iter < MAX_ITERATIONS; iter += 1.0) {
        dist = sceneSDF(ray.origin + t * ray.direction, false);

        if (dist < EPSILON) {
            hitSomething = 1;
            break;
        } else if (t > SCOPE) {
            break;
        }
        t += dist;
    }

    return MarchResult(t, hitSomething, sceneBSDF(ray.origin + t * ray.direction));
}

// The larger the DISTORTION, the smaller the glow
const float DISTORTION = 0.2;
// The higher GLOW is, the smaller the glow of the subsurface scattering
const float GLOW = 6.0;
// The higher the BSSRDF_SCALE, the brighter the scattered light
const float BSSRDF_SCALE = 3.0;
// Boost the shadowed areas in the subsurface glow with this
const float AMBIENT = 0.0;
// Toggle this to affect how easily the subsurface glow propagates through an object
#define ATTENUATION 1

float subsurface(vec3 lightDir, vec3 normal, vec3 viewVec, float thickness) {
    vec3 scatteredLightDir = lightDir + normal * DISTORTION;
    float lightReachingEye = pow(clamp(dot(viewVec, -scatteredLightDir), 0.0, 1.0), GLOW) * BSSRDF_SCALE;
    float attenuation = 1.0;
    #if ATTENUATION
    attenuation = max(0.0, dot(normal, lightDir) + dot(viewVec, -lightDir));
    #endif
    float totalLight = attenuation * (lightReachingEye + AMBIENT) * thickness;
    return totalLight;
}

// Adjust these to alter where the subsurface glow shines through and how brightly
const float FIVETAP_K = 2.0;
const float AO_DIST = 0.085;

float fiveTapAO(vec3 p, vec3 n, float k) {
    float aoSum = 0.0;
    for(float i = 0.0; i < 5.0; ++i) {
        float coeff = 1.0 / pow(2.0, i);
        aoSum += coeff * (i * AO_DIST - sceneSDF(p + n * i * AO_DIST, false));
    }
    return 1.0 - k * aoSum;
}

void main()
{
    Ray ray = rayCast();
    MarchResult result = raymarch(ray);
    BSDF bsdf = result.bsdf;
    bsdf.thinness = fiveTapAO(bsdf.pos, ray.direction, FIVETAP_K);
    vec3 pos = ray.origin + result.t * ray.direction;
    float sub = subsurface(ray.direction, bsdf.nor, -ray.direction, bsdf.thinness);
    vec3 diffuseIrradiance = texture(u_DiffuseIrradianceMap, ray.direction).rgb;
    vec3 color = metallic_plastic_LTE(bsdf, -ray.direction);
    color += (1 - bsdf.metallic) * sub * diffuseIrradiance * bsdf.albedo;

    // Reinhard operator to reduce HDR values from magnitude of 100s back to [0, 1]
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    out_Col = vec4(color, result.hitSomething > 0 ? 1. : 0.);
}

 
