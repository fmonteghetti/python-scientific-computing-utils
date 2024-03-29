/***********************************
* Disk of radius R meshed in (x,y)
/***********************************/


If (!Exists(R)) R=1; EndIf // radius circle

If (!Exists(lc)) lc=1/10; EndIf // char. length

	// disk center
xc = 0; 
yc = 0;

SetFactory("OpenCASCADE");
Circle(1) = {xc, yc, 0, R, 0, 2*Pi};

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

Curve Loop(1) = {1};

Plane Surface(1) = {1};
Physical Surface ("Omega") = {1};
Physical Curve ("Gamma-R") = {1};
