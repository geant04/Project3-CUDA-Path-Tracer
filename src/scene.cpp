#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

using namespace std;
using json = nlohmann::json;

#define DEBUGGING_LOADING 0

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently

        newMaterial.roughness = 1.0f;
        newMaterial.metallic = 0.0f;
        newMaterial.hasReflective = false;
        newMaterial.hasRefractive = false;

        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.metallic = p["METALLIC"];
            newMaterial.hasReflective = true;
        }
        else if (p["TYPE"] == "Transmissive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.hasRefractive = true;
            newMaterial.roughness = p["ROUGHNESS"];
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];

    vector<json> models = {};

    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;

        // Meshes need to be processed separately
        if (type == "mesh")
        {
            models.push_back(p);
            continue;
        }

        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }

    // process mesh info
    for (const auto& model : models)
    {
        processModel(geoms, model, MatNameToID);
    }

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}


// copy and pasted from an example: https://github.com/syoyo/tinygltf/blob/release/examples/basic/main.cpp
bool loadModel(tinygltf::Model &model, const char *filename) {
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

  bool res = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
  if (!warn.empty()) {
    std::cout << "WARN: " << warn << std::endl;
  }

  if (!err.empty()) {
    std::cout << "ERR: " << err << std::endl;
  }

  if (!res)
    std::cout << "Failed to load glTF: " << filename << std::endl;
  else
    std::cout << "Loaded glTF: " << filename << std::endl;

  return res;
}

void Scene::processTriangle(vector<Geom> &geoms, const Geom &triangle)
{
    geoms.push_back(triangle);
    return;
}

void Scene::processModel(vector<Geom> &geoms, const json &jsonModel, std::unordered_map<std::string, uint32_t> &MatNameToID)
{
    // ok yeah somehow we process the mesh here or some bullshit idk
    std::string glbNameString = jsonModel["FILE_PATH"].get<std::string>();
    const char *glbName = glbNameString.c_str();

    std::cout << "Today we will process: " << glbName << std::endl;

    tinygltf::Model model;
    if (!loadModel(model, glbName)) 
    {
        return;
    }
    
    // Model loading time. Time to print out information about the model
    for (const auto &mesh: model.meshes)
    {
        for (const auto &primitive : mesh.primitives)
        {
            if (primitive.indices < 0)
            {
                continue;
            }

            tinygltf::Accessor indexAccessor = model.accessors[primitive.indices];
            tinygltf::BufferView indexBufferView = model.bufferViews[indexAccessor.bufferView];
            tinygltf::Buffer indexBuffer = model.buffers[indexBufferView.buffer];

            // byte data is based on componentType in accessors. Assume right now we are using 5123, which is ushort16
            uint16_t* indexData = reinterpret_cast<uint16_t*>(
                indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset
                );


            std::map<string, int>::const_iterator it = primitive.attributes.find("POSITION");

            // Searched through the enitre iterator and found nothing - no positions, in other words
            if (it == primitive.attributes.end())
            {
                continue;
            }

            tinygltf::Accessor positionAccessor = model.accessors[it->second];
            tinygltf::BufferView positionBufferView = model.bufferViews[positionAccessor.bufferView];
            tinygltf::Buffer positionBuffer = model.buffers[positionBufferView.buffer];

            float* positionData = reinterpret_cast<float*>(
                positionBuffer.data.data() + positionBufferView.byteOffset + positionAccessor.byteOffset
                );

            for (int i = 0; i < indexAccessor.count; i += 3)
            {
                // This is an index we can use directly into our position data.
                // Similarly, we can use this for our normals, etc.
                int i0 = indexData[i];
                int i1 = indexData[i + 1];
                int i2 = indexData[i + 2];

                glm::vec3 v1 = glm::vec3(positionData[i0 * 3], positionData[i0 * 3 + 1], positionData[i0 * 3 + 2]);
                glm::vec3 v2 = glm::vec3(positionData[i1 * 3], positionData[i1 * 3 + 1], positionData[i1 * 3 + 2]);
                glm::vec3 v3 = glm::vec3(positionData[i2 * 3], positionData[i2 * 3 + 1], positionData[i2 * 3 + 2]);

                // Calculate a normal
                glm::vec3 edge1 = v2 - v1;
                glm::vec3 edge2 = v3 - v1;
                glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

                const auto& trans = jsonModel["TRANS"];
                glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);

                v1 += translation;
                v2 += translation;
                v3 += translation;

#if DEBUGGING_LOADING
                glm::vec3 centroid = (v1 + v2 + v3) / 3.0f;

                std::cout << "normal of face " << (i+1)/3 << ": " << normal.x << ", " << normal.y << ", " << normal.z << std::endl;
                std::cout << "centroid of face " << (i+1)/3 << ": " << centroid.x << ", " << centroid.y << ", " << centroid.z << std::endl;
#endif 
                int materialID = MatNameToID[jsonModel["MATERIAL"]];

                triangles.push_back(Triangle{ v1, v2, v3, normal, materialID });
            }
        }
    }

    return;
}