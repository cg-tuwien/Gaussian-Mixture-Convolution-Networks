import pyglet
import torch
import numpy
import trimesh
import trimesh.viewer.windowed
import trimesh.proximity
from pcfitting import data_loading

def project_points_to_plane(points: numpy.array, plane_normal: numpy.array, plane_point: numpy.array):
    vs = plane_point - points
    # distance to plane
    d_plane = numpy.matmul(vs, plane_normal)
    # projected point
    proj_points = points + plane_normal*numpy.tile(d_plane, (3,1)).transpose()
    return proj_points

def project_to_mesh(pc: torch.Tensor, mesh: trimesh.Trimesh) -> torch.Tensor:
    # nfaces = mesh.faces.shape[0]
    # pointfacedistances = numpy.zeros((pc.shape[0], nfaces))
    # for f in range(nfaces):
    #     if not (mesh.face_normals[f] == [0, 0, 0]).all():
    #         pointfacedistances[:, f] = trimesh.points.point_plane_distance(pc.cpu(), mesh.face_normals[f], mesh.vertices[mesh.faces[f, 0]])
    # nearestface = pointfacedistances.argmin(axis=1)
    # projected = numpy.zeros((pc.shape[0], pc.shape[1]))
    # for f in range(nfaces):
    #     if not (mesh.face_normals[f] == [0, 0, 0]).all():
    #         relevantpointidxs = (nearestface == f).nonzero()[0]
    #         if relevantpointidxs.shape[0] > 0:
    #             #projected[relevantpointidxs] = trimesh.points.project_to_plane(pc[relevantpointidxs].cpu(), mesh.face_normals[f], mesh.vertices[mesh.faces[f, 0]], return_planar=False)
    #             projected[relevantpointidxs] = project_points_to_plane(pc[relevantpointidxs].cpu().numpy(), mesh.face_normals[f], mesh.vertices[mesh.faces[f, 0]])
    # return torch.from_numpy(projected).cuda()
    query = trimesh.proximity.ProximityQuery(mesh)
    closest, distance, triangle_id = query.on_surface(pc.cpu().numpy())
    return torch.from_numpy(closest).cuda()

#test = trimesh.points.project_to_plane(numpy.array([[3, 2, 4]]), numpy.array([0, 0, -1]), numpy.array([1, 1, 1]), return_planar=False)

#pcfile = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\resampled\210306-01-EmEckPre\fpsmax\bed_0001.off.gma.ply.off"
for i in range(2):
    pcfile = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\resampled\n100000\210306-01-EmEckPre\EMfps\toilet_0001.off.gma.ply.off"
    pc = data_loading.load_pc_from_off(pcfile)
    meshfile = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\models\bed_0001.off"
    mesh = trimesh.load_mesh(meshfile)
    projected = project_to_mesh(pc[0], mesh)
    # data_loading.write_pc_to_off(pcfile + ".projected.off", projected)
    # scene = mesh.scene()
    # scene.add_geometry(trimesh.points.PointCloud(projected.cpu()))
    # viewer = trimesh.viewer.windowed.SceneViewer(scene, start_loop=False)
    # viewer.toggle_culling()
    # pyglet.app.run()