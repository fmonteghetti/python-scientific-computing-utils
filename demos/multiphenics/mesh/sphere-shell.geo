/***********************************
* Spherical shell
/***********************************/

If (!Exists(R_i)) R_i=0.5; EndIf // inner radius

If (!Exists(R_o)) R_o=1; EndIf // outer radius

If (!Exists(lc)) lc=1/5; EndIf // char. length

SetFactory("OpenCASCADE");

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

Sphere(1) = {0, 0, 0, R_i, -Pi/2, Pi/2, 2*Pi};
Sphere(2) = {0, 0, 0, R_o, -Pi/2, Pi/2, 2*Pi};

BooleanDifference(3) = { Volume{2}; Delete; }{ Volume{1}; Delete; };

Physical Volume("Omega") = {3};
Physical Surface("Gamma-Outer") = {1};
Physical Surface("Gamma-Inner") = {2};
