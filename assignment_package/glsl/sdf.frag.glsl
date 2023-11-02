
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

