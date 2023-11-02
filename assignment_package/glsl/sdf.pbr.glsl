
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
