#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateHemisphereDirection(glm::vec3 normal, float cosTheta, float phi)
{
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return cosTheta * normal
        + cos(phi) * sinTheta * perpendicularDirection1
        + sin(phi) * sinTheta * perpendicularDirection2;
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float cosTheta = sqrt(u01(rng)); // cos(theta)
    float phi = u01(rng) * TWO_PI;

    return calculateHemisphereDirection(normal, cosTheta, phi);
}

__host__ __device__ glm::vec3 calculateWalterGGXSampling(
    glm::vec3 normal,
    float roughness,
    thrust::default_random_engine &rng)
{
    //// https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
    //// Eq. 35, 36 for GGX sampling distribution
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float u0 = u01(rng);
    float u1 = u01(rng);

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float cosTheta = sqrt((1.0f - u0) / (u0 * (alpha2 - 1.0f) + 1.0f)); 
    float sinTheta = sqrt(1 - cosTheta * cosTheta);
    float phi = TWO_PI * u1;

    return calculateHemisphereDirection(normal, cosTheta, phi);
}

// Have to include this stupid normal so the intellisense doesn't go crazy
__host__ __device__ glm::vec3 calculateRandomDirectionInSphere(float u1, float u2)
{
    float z = 2.0f * u1 - 1.0f;
    float phi = TWO_PI * u2;
    float r = sqrt(1.0f - z * z);

    return glm::vec3(r * cos(phi), r * sin(phi), z);
}

__host__ __device__ float isotropicSampleDistance(
    float tFar, 
    float scatteringCoefficient, 
    float &weight,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float distance = -logf(u01(rng)) / scatteringCoefficient;

    if (distance >= tFar)
    {
        // pdf assignment that unfortunately doesn't exist
        return tFar;
    }

    weight = exp(-scatteringCoefficient * distance);
    return distance;
}

__host__ __device__ glm::vec3 isotropicTransmission(glm::vec3 absorptionCoefficient, float distance)
{
    return exp(-absorptionCoefficient * distance);
}

__host__ __device__ glm::vec3 isotropicSampleScatterDirection(float u1, float u2)
{
    return calculateRandomDirectionInSphere(u1, u2);
}

// MASSIVE RAY SAMPLING FUNCTION!!!!!!!!!!!!!!!
__host__ __device__ void sampleRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 inDirection = glm::normalize(pathSegment.ray.direction);
    glm::vec3 outDirection;
    glm::vec3 halfVector = glm::normalize(inDirection + normal);

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Using Joe Schutte's Disney implementation for this
    float metallicWeight = m.metallic;
    float diffuseWeight = (1.0f - metallicWeight);
    float specularWeight = metallicWeight + diffuseWeight;

    float invSumWeight = 1.0f;

    float pDiffuse = diffuseWeight * invSumWeight;
    // float pSpecular = specularWeight * invSumWeight;

    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.color, m.metallic);
    float pSpecular = glm::clamp((F0.x + F0.y + F0.z) / 3.0f, 0.02f, 0.98f);

    
    // Distance value used by transmission when within a medium
    float t = glm::length(intersect - pathSegment.ray.origin);
    float p = u01(rng);
    float epsilon = 0.0005f;
    glm::vec3 brdf = m.color;

    glm::vec3 wo = -inDirection;
    glm::vec3 wi;
    glm::vec3 intersectOffset;
    glm::vec3 diffuse_wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    glm::vec3 diffuseNormal = dot(normal, diffuse_wi) < 0.0f ? -diffuse_wi : diffuse_wi;

    if (pathSegment.medium != VACUUM)
    {
        float scatteringDistance = 1.0f;
        float scatteringCoefficient = 1.0f / scatteringDistance;
        float weight = 1.0f;
        float distance = isotropicSampleDistance(t, scatteringCoefficient, weight, rng);

        float absorptionAtDistance = 5.0f;
        glm::vec3 absorptionColor = glm::vec3(0.85f, 0.35f, 0.35f);
        glm::vec3 absorptionCoefficient = -log(absorptionColor) / absorptionAtDistance;

        // RichieSams has us create a "scatter event".
        if (distance < t)
        {
            glm::vec3 transmission = isotropicTransmission(absorptionCoefficient, distance);
            pathSegment.color *= transmission * weight;

            pathSegment.ray.origin += pathSegment.ray.direction * distance;
            pathSegment.ray.direction = isotropicSampleScatterDirection(u01(rng), u01(rng));
        }
        // No scatter event, we've hit the surface and we're ready to leave!
        else
        {
            glm::vec3 transmission = exp(-absorptionCoefficient * t);
            pathSegment.color *= transmission;

            // No need to change ray direction, should be ok
            pathSegment.medium = VACUUM;
            pathSegment.ray.origin = intersect + normal * 0.001f;
        }

        return;
    }

    if (m.hasReflective)
    {
        // Specular GGX
        glm::vec3 microNormal = glm::normalize(calculateWalterGGXSampling(normal, m.roughness, rng));
        pathSegment.microNormal = microNormal;

        glm::vec3 specularDir = glm::reflect(inDirection, microNormal);

        glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.color, m.metallic);
        float avgF0 = glm::clamp((F0.x + F0.y + F0.z) / 3.0f, 0.02f, 0.98f);
        bool isSpecularBounce = p < avgF0;

        wi = glm::mix(
            diffuse_wi, 
            specularDir, 
            isSpecularBounce);
        brdf = glm::mix(
            diffuseBRDF(wo, wi, normal, m) / (1.0f - avgF0), 
            specularBRDF(wo, wi, normal, microNormal, m) / avgF0, 
            isSpecularBounce);
        intersectOffset = wi * epsilon;
    }
    else if (m.subsurface > 0.0f)
    {
        // Refract into material thing... idk man
        wi = -diffuse_wi;

        pathSegment.medium = ISOTROPIC;
        pathSegment.ray.origin = intersect + wi * 0.001f;
        pathSegment.ray.direction = wi;

        return;
    }
    else if (m.hasRefractive)
    {
        glm::vec3 diffuseWi = glm::mix(normal, diffuseNormal, m.roughness);

        float cosThetaI = dot(normal, wo);        
        float etaA = 1.0f;
        float etaB = 1.55f;

        float rand = u01(rng);

        glm::vec3 R0 = glm::vec3((etaA - etaB) / (etaA + etaB));
        R0 = R0 * R0;
        glm::vec3 F = fresnelSchlick(R0, abs(cosThetaI));

        // no tint, do this for now
        brdf = glm::vec3(1.0f);

        if (rand < length(F))
        {
            wi = glm::reflect(glm::normalize(-wo), normal);
            wi = glm::mix(wi, diffuseWi, m.roughness);
            
            // awesome artificial roughness trick from seb. lague
            float cosTheta = abs(dot(normal, wi));
            glm::vec3 R0 = glm::vec3((etaA - etaB) / (etaA + etaB));
            R0 = R0 * R0;
            glm::vec3 F = fresnelSchlick(R0, abs(cosThetaI));

            // brdf = 2.0f * F;
        }
        else
        {
            // Transmissive material, use the specularBTDF
            float cosThetaI = dot(normal, wo);
            bool entering = cosThetaI > 0.0f;

            float eta = etaA / etaB;
            float iorRatio = (entering) ? eta : 1.0f / eta;

            wi = glm::refract(inDirection, (entering) ? normal : -normal, iorRatio);
            wi = glm::mix(wi, (entering) ? -diffuseNormal : diffuseNormal, m.roughness);

            if (length(wi) < 0.01f)
            {
                wi = glm::reflect(inDirection, normal);
            }

            // float F = fresnelDielectric(abs(cosThetaI), etaA, etaB);
            float cosTheta = abs(dot(normal, wi));
            glm::vec3 R0 = glm::vec3((etaA - etaB) / (etaA + etaB));
            R0 = R0 * R0;
            glm::vec3 F = fresnelSchlick(R0, abs(cosThetaI));

            brdf = glm::vec3(1.0f);

            if (!entering) 
            {
                float scatterDistance = sqrt(t);
                float absorptionMultiplier = 3.0f;
                glm::vec3 absorptionTint = exp(-scatterDistance * m.color * absorptionMultiplier);

                brdf *= absorptionTint;
            }

            // brdf = glm::vec3(1.0f - F);
        }

        // Figure out why we need this hack... I suspect something about intersecting at a weird/shallow angle
        // epsilon = 0.05f;
    }
    else
    {
        // Sample diffuse
        wi = diffuse_wi;
        brdf = diffuseBRDF(wo, wi, normal, m);
    }

    // Assign wi
    pathSegment.ray.direction = wi;

    // Assign intersect for the next bounce
    pathSegment.ray.origin = intersect + wi * epsilon; // wi * epsilon;

    // Yeah
    pathSegment.color *= brdf;

    pathSegment.remainingBounces -= 1;
}

__host__ __device__ float GGXDistribution(float alpha, float cosTheta)
{
    // https://agraphicsguynotes.com/posts/sample_microfacet_brdf/#one-extra-step
    float alpha2 = alpha * alpha;
    float cos2Theta = cosTheta * cosTheta;
    float denom = (alpha2 - 1.0f) * cos2Theta + 1.0f;

    return alpha2 / (PI * denom * denom);
}

__host__ __device__ float SmithGGX(
    float nDotI,
    float nDotO,
    float a2
)
{
    // Based on implementation from https://schuttejoe.github.io/post/ggximportancesamplingpart1/,
    // https://media.gdcvault.com/gdc2017/Presentations/Hammon_Earl_PBR_Diffuse_Lighting.pdf, this too maybe.
    // This combines SmithGGX(i, m) * SmithGGX(o, m)
    float denomIn = nDotI * sqrt(a2 + (1.0f - a2) * (nDotI * nDotI));
    float denomOut = nDotO * sqrt(a2 + (1.0f - a2) * (nDotO * nDotO));

    return 2.0f * nDotI * nDotO / (denomIn + denomOut);
}

__host__ __device__ glm::vec3 fresnelSchlick(glm::vec3 F0, float cosTheta)
{
    return F0 + (glm::vec3(1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
}

// Ripped straight from PBRT 3ed, thanks Google
__host__ __device__ float fresnelDielectric(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    // Potentially swap indices of refraction
    if(cosThetaI < 0.0f) {
        std::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    // Compute cosTheta using Snell's law
    float sinThetaI = glm::sqrt(glm::max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Check for total internal reflection
    if(sinThetaT >= 1) {
        return 1;
    }

    float cosThetaT = glm::sqrt(glm::max(0.0f, 1.0f - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__host__ __device__ glm::vec3 specularBTDF(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 normal,
    const Material &m
)
{
    float cosThetaWo = dot(normal, wo);
    bool entering = cosThetaWo > 0;

    return glm::vec3(0.0f, 1.0f, 0.0f);
}


__host__ __device__ glm::vec3 specularBRDF(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 normal,
    glm::vec3 microNormal,
    const Material &m
)
{
    glm::vec3 half = glm::normalize(wo + wi);
        
    float nDotH = glm::max(dot(normal, half), 0.0f);
    float nDotI = glm::max(dot(normal, wo), 0.0f);
    float nDotO = glm::max(dot(normal, wi), 0.0f);
    float mDotI = glm::max(dot(microNormal,wi), 0.0f);

    if (dot(wi, microNormal) < 0.0f)
    {
        mDotI = -mDotI;
    }

    float G = SmithGGX(nDotI, nDotO, m.roughness);
       
    glm::vec3 albedo = m.color;
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, m.metallic);
    
    glm::vec3 metallicF = fresnelSchlick(F0, glm::max(dot(wi, half), 0.0f));
    // I hard coded the IOR of plastic,which is 1.460, into the third arg of fresnelDielectric
    glm::vec3 dielectricF = glm::clamp(glm::vec3(fresnelDielectric(mDotI, 1.0f, 1.460f)), 0.0f, 1.0f); 

    glm::vec3 F = glm::mix(dielectricF, metallicF, m.metallic);

    // Adapted this from Schutte's specular BRDF simplification
    // I'm assuming the D term isn't here because of cut terms from fully evaluating GGX,
    // but I'm confused about why the denominator just doesn't exist in this implementation.
    return F * G; // glm::vec3(G);
}

__host__ __device__ glm::vec3 diffuseBRDF(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 normal,
    const Material &m
)
{
    return m.color; //glm::clamp(m.color, glm::vec3(0.0f), glm::vec3(1.0f));
}