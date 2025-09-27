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

#define DEBUG_GLTF_LOADING 0
#define DEBUG_BVH_BUILDING 0

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
        newMaterial.subsurface = 0.0f;
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
        else if (p["TYPE"] == "Subsurface")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.subsurface  = p["SUBSURFACE"];
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
        std::cout << "tris loaded: " << triangles.size() << std::endl;
    }

    // BVH !!!!!!! so important...
    if (triangles.size() > 0)
    {
        buildBVH();
    }

    //////////////////////////////////////////
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

                const auto& trans = jsonModel["TRANS"];
                glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);

                v1 += translation;
                v2 += translation;
                v3 += translation;

                // Calculate centroid, needed for BVH building
                glm::vec3 centroid = (v1 + v2 + v3) / 3.0f;

#if DEBUG_GLTF_LOADING
                if (i < 2)
                {
                // Calculate a normal
                glm::vec3 edge1 = v2 - v1;
                glm::vec3 edge2 = v3 - v1;
                glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

                std::cout << "normal of face " << (i+1)/3 << ": " << normal.x << ", " << normal.y << ", " << normal.z << std::endl;
                std::cout << "centroid of face " << (i+1)/3 << ": " << centroid.x << ", " << centroid.y << ", " << centroid.z << std::endl;
                }
#endif 
                int materialID = MatNameToID[jsonModel["MATERIAL"]];

                triangles.push_back(Triangle{ v1, v2, v3, centroid, materialID });
            }
        }
    }
    
    return;
}

void Scene::buildBVH()
{
    // Triangle centroids have already been computed by this point
    int rootIdx = 0;
    int nodesUsed = 1;

    int n = triangles.size();
    bvhNodes = std::vector<BVHNode>(n);

    // Layer of index indirection
    for (int i = 0; i < n; i++)
    {
        triangleIndices.push_back(i);
    }

    // Root node idx = 0
    BVHNode& rootNode = bvhNodes[rootIdx];
    rootNode.leftChild = 0;
    rootNode.rightChild = 0;
    rootNode.firstIdx = 0;
    rootNode.prims = n;

    updateBVHBounds(rootIdx);
    splitBVHNode(rootIdx, nodesUsed);
}

void Scene::updateBVHBounds(int nodeIdx)
{
    BVHNode &node = bvhNodes[nodeIdx];

    // Root node min/max AABB bounds calculation
    node.aabbMin = glm::vec3(1e30f);
    node.aabbMax = glm::vec3(-1e30f);

    for (int i = node.firstIdx; i < node.firstIdx + node.prims; i++)
    {
        int leafIdx = triangleIndices[i];
        const Triangle &prim = triangles[leafIdx];

        node.aabbMin = glm::min(node.aabbMin, prim.v1);
        node.aabbMin = glm::min(node.aabbMin, prim.v2);
        node.aabbMin = glm::min(node.aabbMin, prim.v3);
        
        node.aabbMax = glm::max(node.aabbMax, prim.v1);
        node.aabbMax = glm::max(node.aabbMax, prim.v2);
        node.aabbMax = glm::max(node.aabbMax, prim.v3);
    }
}

void Scene::splitBVHNode(int nodeIdx, int &nodesUsed)
{
#if DEBUG_BVH_BUILDING
    std::cout << "splitting node: " << nodeIdx << std::endl;
#endif
    BVHNode &node = bvhNodes[nodeIdx];

    if (node.prims <= 2) 
    {
#if DEBUG_BVH_BUILDING
        std::cout << "stopping, building leaf " << nodeIdx << ", children: " << node.prims << std::endl;
#endif
        return;
    }
    // Determine axis and position of split plane
    glm::vec3 extent = node.aabbMax - node.aabbMin;

    // Fancy way of finding the max of XYZ. Return axis in form of 0,1,2
    int axis = (extent.z > extent.y) ? 2 : (extent.y > extent.x) ? 1 : 0;
    
    // Split group of prim in 2 halves w split plane
    // Notice extent[axis] * 0.5f -- this is to split in HALF. May change later
    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;

    int leftPrimIndex = node.firstIdx;
    int rightPrimIndex = leftPrimIndex + node.prims - 1;

    while (leftPrimIndex <= rightPrimIndex)
    {
        int leftTriangleIndex = triangleIndices[leftPrimIndex];

        // Is left?
        if (triangles[leftTriangleIndex].centroid[axis] < splitPos)
        {
            leftPrimIndex++;
        }
        // Is right?
        else
        {
            std::swap(triangleIndices[leftPrimIndex], triangleIndices[rightPrimIndex]);
            rightPrimIndex--;
        }
    }

    int leftPrims = leftPrimIndex - node.firstIdx;
    int rightPrims = node.prims - leftPrims;

    // In theory, one of the boxes can be completely empty. This will handle that
    if (leftPrims == 0 || rightPrims == 0) {
#if DEBUG_BVH_BUILDING
        std::cout << "degenerate split, stopping and making idx " << nodeIdx << " a leaf!" << std::endl;
#endif
        return;
    }

    // Create children for each half
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;
    
    node.leftChild = leftChildIdx;
    bvhNodes[leftChildIdx].firstIdx = node.firstIdx;
    bvhNodes[leftChildIdx].prims = leftPrims;

    node.rightChild = rightChildIdx;
    bvhNodes[rightChildIdx].firstIdx = leftPrimIndex;
    bvhNodes[rightChildIdx].prims = node.prims - leftPrims;

    // Make sure to reset it so that we can use this to identify leaves.
    node.prims = 0;

    // Update bounds
    updateBVHBounds(leftChildIdx);
    updateBVHBounds(rightChildIdx);

    // Recurse into children
    splitBVHNode(leftChildIdx, nodesUsed);
    splitBVHNode(rightChildIdx, nodesUsed);
}