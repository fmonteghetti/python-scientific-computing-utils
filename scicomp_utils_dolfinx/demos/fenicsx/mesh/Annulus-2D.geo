/***********************************
* Annulus of radiuses R0 and R1.
/***********************************/

If (!Exists(R0)) R0=0.5; EndIf // smaller circle
If (!Exists(R1)) R1=1.0; EndIf // greater circle
If (!Exists(lc)) lc=(R1-R0)/2; EndIf // char. length

	// disk center
xc = 0; 
yc = 0;

SetFactory("OpenCASCADE");
Circle(1) = {xc, yc, 0, R0, 0, 2*Pi};
Circle(2) = {xc, yc, 0, R1, 0, 2*Pi};

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
// Mesh.ElementOrder = 2;

// Two closed circles
Curve Loop(1) = {1};
Curve Loop(2) = {2};

// Surface: annulus
Plane Surface(1) = {1,2};
Physical Surface ("Omega") = {1};
Physical Curve ("Gamma-0") = {1};
Physical Curve ("Gamma-1") = {2};