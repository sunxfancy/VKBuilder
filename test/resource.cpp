#include <string>
#include <iostream>

#include "vkbuilder.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

using namespace std;
using namespace vkb;

TextureImage2D* loadImage(Device& device, vk::CommandPool commandPool, vk::Queue queue, string path) {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    vk::DeviceSize imageSize = texWidth * texHeight * 4;
    auto image = new TextureImage2D(device, texWidth, texHeight);
    image->upload(commandPool, queue, pixels, imageSize);
    stbi_image_free(pixels);
    return image;
}

static std::string GetFilePathExtension(const std::string &FileName) {
  if (FileName.find_last_of(".") != std::string::npos)
    return FileName.substr(FileName.find_last_of(".") + 1);
  return "";
}

tinygltf::Model* loadModel(string path) {
    tinygltf::Model* model = new tinygltf::Model();
    tinygltf::TinyGLTF gltf_ctx;

    std::string err;
    std::string warn;
    std::string ext = GetFilePathExtension(path);

    bool ret = false;
    if (ext == "glb") {
        std::cout << "Reading binary glTF" << std::endl;
        // assume binary glTF.
        ret = gltf_ctx.LoadBinaryFromFile(model, &err, &warn, path.c_str());
    } else {
        std::cout << "Reading ASCII glTF" << std::endl;
        // assume ascii glTF.
        ret = gltf_ctx.LoadASCIIFromFile( model, &err, &warn, path.c_str());
    }

    return model;
}