########################################################
## Mesh circle (0,R)
# Mesh 1: unstructured (x,y)
# Mesh 2: unstructured (x,y) + Euler (z,theta)
########################################################

import gmsh
import sys
import math as mt

pi = mt.pi

gmsh.initialize()
gmsh.logger.start()
gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.add("circle")

R = 1
lc = 1e-1
p_center = gmsh.model.occ.addPoint(0, 0, 0, lc)
p_start = gmsh.model.occ.addPoint(R, 0, 0, lc)
l_circle = gmsh.model.occ.addCircle(0, 0, 0, R)
l_circleLoop = gmsh.model.occ.addCurveLoop([l_circle])
s_disk = gmsh.model.occ.addPlaneSurface([l_circleLoop])
gmsh.model.occ.synchronize()
tmp = gmsh.model.addPhysicalGroup(1, [l_circleLoop])
gmsh.model.setPhysicalName(1, tmp, "Boundary")

tmp = gmsh.model.addPhysicalGroup(2, [s_disk])
gmsh.model.setPhysicalName(2, tmp, "Disk")

# Check order in GUI using Mesh\Inspect on boundary triangles
gmsh.model.mesh.setOrder(2)
gmsh.option.setNumber("Mesh.ElementOrder", 2)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
field1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumber(field1, "EdgesList", l_circle)
gmsh.model.mesh.field.setNumber(field1, "NodesList", p_start)
gmsh.model.mesh.field.setNumber(field1, "NNodesByEdge", 1e4)
field2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field2, "IField", field1)
gmsh.model.mesh.field.setNumber(field2, "DistMin", 0)
gmsh.model.mesh.field.setNumber(field2, "DistMax", R)
gmsh.model.mesh.field.setNumber(field2, "LcMin", 1e-3)
gmsh.model.mesh.field.setNumber(field2, "LcMax", 1e-1)
gmsh.model.mesh.field.setNumber(field2, "Sigmoid", True)
gmsh.model.mesh.field.setAsBackgroundMesh(field2)
gmsh.model.mesh.generate(2)

# ... and save it to disk
gmsh.write("t1.msh")

# Inspect the log:
log = gmsh.logger.get()
gmsh.logger.stop()

print("Logger has recorded " + str(len(log)) + " lines")
print(*log, sep="\n")


gmsh.finalize()
