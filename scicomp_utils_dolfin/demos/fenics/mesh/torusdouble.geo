/***********************************
* Torus split in two domains.
/***********************************/

If (!Exists(R_ma)) R_ma=1; EndIf // major radius

If (!Exists(R_mi)) R_mi=0.5; EndIf // minor radius

If (!Exists(lc)) lc=1/5; EndIf // char. length

SetFactory("OpenCASCADE");

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

//+
Torus(1) = {0, 0, 0, R_ma, R_mi, Pi};
//+
Symmetry {0, 1, 0, 0} {
  Volume{1}; 
}
//+
Torus(2) = {0, 0, 0, R_ma, R_mi, Pi};
    // Conforming mesh
Coherence;
//+
Physical Surface("Gamma-T") = {4};
//+
Physical Surface("Gamma-B") = {1};
//+
Physical Volume("Omega-T") = {1};

Physical Volume("Omega-B") = {2};
//+
//+
Physical Surface("Gamma-Int") = {2, 3};
