#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle &triangle,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &intersectionNormal,
    bool &outside)
{
    glm::vec3 v1 = triangle.v1;
    glm::vec3 v2 = triangle.v2;
    glm::vec3 v3 = triangle.v3;

    glm::vec3 v2v1 = v2 - v1;
    glm::vec3 v3v1 = v3 - v1;
    glm::vec3 normal = glm::normalize(glm::cross(v2v1, v3v1));

    float nDotDir = dot(normal, r.direction);

    // Scratchapixel ray/plane intersection
    // Keep backface triangle intersection, needed for transmission!
    // removing v1 - centroid to just v1 causes... a memory access error??
    float D = -dot(normal, (v1));
    float t = -(dot(normal, r.origin) + D) / dot(normal, r.direction);

    glm::vec3 intersectP = r.origin + t * r.direction;
    glm::vec3 tangentToNor;

    // Edge test for (v2 - v1), 
    // Cross of calculated tan to nor < 0 means p is on right side of edge.
    tangentToNor = glm::cross((v2 - v1), (intersectP - v1));
    if (dot(tangentToNor, normal) < 0.0f)
    {
        return -1.0f;
    }

    // Edge test for (v3 - v2),
    tangentToNor = glm::cross((v3 - v2), (intersectP - v2));
    if (dot(tangentToNor, normal) < 0.0f)
    {
        return -1.0f;
    }

    // Edge test for (v1 - v3)
    tangentToNor = glm::cross((v1 - v3), (intersectP - v3));
    if (dot(tangentToNor, normal) < 0.0f)
    {
        return -1.0f;
    }

    outside = dot(normal, r.direction) > 0.0f;
    intersectionPoint = intersectP;
    intersectionNormal = normal;

    return t;
}

__host__ __device__ bool aabbIntersectionTest(
    const Ray &r,
    float curr_tmin,
    glm::vec3 aabbMin,
    glm::vec3 aabbMax)
{
    // Copying aabb collision logic from Jacco Bikker's blog for now. Need to study it later.
    float tx1 = (aabbMin.x - r.origin.x) / r.direction.x;
    float tx2 = (aabbMax.x - r.origin.x) / r.direction.x;
    float tmin = glm::min(tx1, tx2);
    float tmax = glm::max(tx1, tx2);

    float ty1 = (aabbMin.y - r.origin.y) / r.direction.y;
    float ty2 = (aabbMax.y - r.origin.y) / r.direction.y;
    tmin = glm::max(tmin, glm::min(ty1, ty2));
    tmax = glm::min(tmax, glm::max(ty1, ty2));

    float tz1 = (aabbMin.z - r.origin.z) / r.direction.z;
    float tz2 = (aabbMax.z - r.origin.z) / r.direction.z;
    tmin = glm::max(tmin, glm::min(tz1, tz2));
    tmax = glm::min(tmax, glm::max(tz1, tz2));

    return tmax >= tmin && tmin < curr_tmin && tmax > 0;
}


__host__ __device__ float bvhIntersectionTest(
    const Ray &r,
    BVHNode* bvhNodes, 
    Triangle* triangles, 
    int* triangleIndices,
    glm::vec3 &intersectionPoint,
    glm::vec3 &intersectionNormal,
    int &minTriIndex,
    float curr_tmin)
{
    // As it stands for our awesome simple cube, there are only 2 nodes.
    // Recursive to iterate logic taken from Sebastian Lague. Why size of 10? I think in theory, our stack is only 2 indices large.
    int nodeIdxStack[64];
    int stackIdx = 0;
    nodeIdxStack[stackIdx++] = 0;

    float t = curr_tmin;
    glm::vec3 tempIsectP = intersectionPoint;
    glm::vec3 tempNormal = intersectionNormal;
    int tempTriIdx = -1;

    while (stackIdx > 0)
    {
        // Pop off stack
        int nodeIdx = nodeIdxStack[--stackIdx];
        BVHNode &node = bvhNodes[nodeIdx];

        if (!aabbIntersectionTest(r, curr_tmin, node.aabbMin, node.aabbMax))
        {
            continue;
        }
        
        // Leaf check. Has no children.
        if (node.prims > 0)
        {
            glm::vec3 dummyIsect;
            glm::vec3 dummyNormal;
            bool dummyOutside;

            for (int i = node.firstIdx; i < node.firstIdx + node.prims; i++)
            {
                int triangleIdx = triangleIndices[i];
                Triangle &triangle = triangles[triangleIdx];

                float tempT = triangleIntersectionTest(triangle, r, dummyIsect, dummyNormal, dummyOutside);
                if (tempT > 0.0f && t > tempT)
                {
                    t = tempT;
                    tempIsectP = dummyIsect;
                    tempNormal = dummyNormal;
                    tempTriIdx = triangleIdx;
                }
            }
        }
        else
        {
            nodeIdxStack[stackIdx++] = node.leftChild;
            nodeIdxStack[stackIdx++] = node.rightChild;
        }
    }

    if (t > 0.0f)
    {
        intersectionPoint = tempIsectP;
        intersectionNormal = tempNormal;
        minTriIndex = tempTriIdx;
        return t;
    }

    return -1.0f;
}