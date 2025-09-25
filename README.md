CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.


## Ge Pathtracer BRDF Model
### Diffuse BRDF + Microfacet Specular GGX

For my BRDF, I use the Cook-Torrance model to simulate diffuse and specular surfaces. I use a uniform random number from 0-1 to sample between the two surfaces, where if probability is less than average F0, I sample diffuse, otherwise I sample specular. This allows dielectrics to sample more diffuse, whereas conductors absorb and require fully specular reflections.

Diffuse sampling uses the uniform cosine-weighted hemisphere sampling for wi, while I use the GGX NDF to sample the specular direction such that roughness now affects the lobe of samples. Upon dividing out BRDF/pdf, the diffuse weight evaluates to just material albedo.

For specular BRDF, I use the GGX microfacet model. Referencing Joe Schutte's article on [sampling with GGX](https://schuttejoe.github.io/post/ggximportancesamplingpart1/) and Walter's paper on weighting samples themselves, we can observe that the returned reflectance can be simplified to F * G, divided by pdf of outgoing rays correctly being in the hemisphere. Through simplification, we just evaluate the specular weight as F * G for now as a "close-enough" approximation, unfortunately. Unless I get some time to figure out how to accurately implement the rest of the model, which would be (D * G * F) / (4.0f * cos_wo * cos_wi).

### Extending the BRDF to include transmissive surfaces
I haven't done this yet, but I plan to extend my BRDF model to also include BTDF, creating an ultimate BSDF model! I aim to also reference Joe Schutte's implement of the Disney BSDF to achieve this.

So while I haven't done this yet, I DID implement BTDF, being able to render a glass ball if it's strictly chosen as such. For a given transmissive surface, I choose to either refract or reflect based on a 50/50 probability.

The refraction allows us to enter the transmissive surface and bend the incoming ray direction by Snell's law, meaning that for a given material we need to classify its entering and exiting IORs. The reflection, on the other hand, is a simple reflect of the incoming ray by the normal. By weighting the power of each refracted and reflected ray by its fresnel Schlick approximation, we are able to have reflections off grazing angles, while most of the ball becomes transmissive.

Using a transmissive term, I believe it's possible to easily include this into the existing BRDF model, simply using another probability weight to choose whether or not for a given materia we use BRDF or BTDF. More on this later if I get to it, but will instead work on gltf loading instead.

## Various Goodies
### Depth of Field
TO DO: put in stuff

### Bokeh(?)

## Mesh Rendering with GLTF
gltf is an awesome model format. It represents a given scene by a tree hierarchy, starting from the scene to nodes with children that have primitives. Our goal is to convert these primitives into triangles that our pathtracing intersection test can detect.

### Using the tinyGLTF loader
Figuring this out right now.


### The non-existent BVH if it works


Feedback
- The sphere intersection normal flipping code should be removed.
- stb_image files should be updated, it might make gltf importing easier for those using tinyGLTF, but also it doesn't hurt to use more updated libraries.