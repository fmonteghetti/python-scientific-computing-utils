// Torus with unstructured mesh

If (!Exists(R_ma)) R_ma=1; EndIf // major radius

If (!Exists(R_mi)) R_mi=0.5; EndIf // minor radius

If (!Exists(lc)) lc=1/5; EndIf // char. length

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

SetFactory("OpenCASCADE");

Torus(1) = {0, 0, 0, R_ma, R_mi, 2*Pi};

Physical Surface("Gamma") = {1};

Physical Volume("Omega") = {1};
