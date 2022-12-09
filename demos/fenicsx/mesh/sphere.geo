/***********************************
* Sphere.
/***********************************/

If (!Exists(R)) R=1; EndIf

If (!Exists(lc)) lc=1/10; EndIf // char. length


SetFactory("OpenCASCADE");

Sphere(1) = {0, 0, 0, R, -Pi/2, Pi/2, 2*Pi};

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

Coherence;

Physical Volume ("Omega") = {1};
Physical Surface ("Gamma") = {1};
