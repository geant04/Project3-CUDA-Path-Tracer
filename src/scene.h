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
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    RenderState state;
};
