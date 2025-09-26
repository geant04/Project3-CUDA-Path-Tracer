#pragma once

#include "sceneStructs.h"
#include <vector>
#include "json.hpp"

using json = nlohmann::json;

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void processModel(
        std::vector<Geom> &geoms, 
        const json &mesh, 
        std::unordered_map<std::string, uint32_t> &MatNameToID);
    void processTriangle(std::vector<Geom> &geoms, const Geom &triangle);

    // TO DO: Move this to its own BVH cpp
    void buildBVH();
    void updateBVHBounds(int nodeIdx);
    void splitBVHNode(int nodeIdx, int &nodesUsed);
public:
    Scene(std::string filename);

    // Mesh loading/BVH shenanigans
    std::vector<Triangle> triangles;
    std::vector<int> triangleIndices;
    std::vector<BVHNode> bvhNodes;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
