/***********************************
* Disk of radius R meshed in (x,y)
/***********************************/


If (!Exists(R)) R=1; // radius circle
EndIf 

If (!Exists(R_TR)) R_TR=0.5; // truncation radius
EndIf 
	// disk center
xc = 0; 
yc = 0;

SetFactory("OpenCASCADE");
Circle(1) = {xc, yc, 0, R_TR, 0, 2*Pi};
Circle(2) = {xc, yc, 0, R, -0.25*Pi, 0.25*Pi};
Circle(3) = {xc, yc, 0, R, 0.25*Pi, (2-0.25)*Pi};

Mesh.CharacteristicLengthMin = 0.5/1;
Mesh.CharacteristicLengthMax = 0.5/1;
//Mesh.ElementOrder = 2;

// Two closed circles
Curve Loop(1) = {1};
Curve Loop(2) = {2,3};

// Surface: disk
Plane Surface(1) = {1};
Physical Surface ("Omega-int") = {1};
// Surface: annulus
Plane Surface(2) = {1,2};
Physical Surface ("Omega-ext") = {2};
Physical Curve ("Gamma-int") = {1};
Physical Curve ("Gamma-D") = {2};
Physical Curve ("Gamma-D2") = {3};

