/***********************************
* Cube
/***********************************/

If (!Exists(Lx)) Lx=1; EndIf
If (!Exists(Ly)) Ly=1; EndIf
If (!Exists(Lz)) Lz=1; EndIf

If (!Exists(lc)) lc=1/10; EndIf // char. length

SetFactory("OpenCASCADE");

//Sphere(1) = {0, 0, 0, R, -Pi/2, Pi/2, 2*Pi};
   
Box(1) = {0,0,0,Lx,Ly,Lz};

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

Coherence;

Physical Volume ("Omega") = {1};
Physical Surface ("Gamma") = {1,2,3,4,5,6};
