import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
import numpy as np
from skimage import measure

if __name__ == '__main__':

    def plot_3d(image, ax, threshold=-300,
                alpha=0.1, face_color=[0.5, 0.5, 1],
                cut=None):
        p = image.transpose(2, 1, 0)

        if cut is not None:
            indices = np.array([[i, j, k] for i in range(p.shape[0])
                                for j in range(p.shape[1])
                                for k in range(p.shape[2])])
            print(indices.shape)

            x, y, z = cut
            remove = np.logical_and(indices[..., 0] > x, indices[..., 1] < y)
            print(remove)
            indices = indices[remove]
            print(indices)

            # p[indices] = threshold - 1
            p[x:, :y, :] = threshold - 1

        print('skimage .....')
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            p, threshold)

        # print(verts, verts.shape)
        # print(faces, faces.shape)

        vert = verts[faces]

        # if cut is not None:
        #     x, y, z = cut
        #     print(vert.shape)
        #     print(vert[0])
        #     # vert[vert[0, ...] > x] = np.Na
        #     print(np.logical_or(vert[..., 0] < x, vert[..., 1] > y))
        #     selected = np.logical_or(vert[..., 0] < x, vert[..., 1] > y)
        #     vert = vert[selected]

        mesh = Poly3DCollection(vert, alpha=alpha)
        # mesh = Poly3DCollection(np.argwhere(p > threshold), alpha=1)
        ls = LightSource(azdeg=225.0, altdeg=45.0)
        normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3), np.sum(normals[face[:], 1]/3), np.sum(normals[face[:], 2]/3))/np.sqrt(
            np.sum(normals[face[:], 0]/3)**2 + np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:], 2]/3)**2)) for face in faces])

        # min shade value
        min = np.min(ls.shade_normals(normalsarray, fraction=1.0))
        # max shade value
        max = np.max(ls.shade_normals(normalsarray, fraction=1.0))
        diff = max-min
        newMin = 0.3
        newMax = 0.95
        newdiff = newMax-newMin

        # Using a constant color, put in desired RGB values here.
        # np.array((255.0/255.0, 54.0/255.0, 57/255.0, 1.0))
        colourRGB = np.array((*face_color, 1.0))

        # The correct shading for shadows are now applied. Use the face normals and light orientation to generate a shading value and apply to the RGB colors for each face.
        rgbNew = np.array(
            [colourRGB*(newMin + newdiff*((shade-min)/diff))
             for shade in ls.shade_normals(normalsarray, fraction=1.0)])

        # mesh.set_facecolor(face_color)
        mesh.set_facecolor(rgbNew)
        ax.add_collection3d(mesh)
        # ax.set_xlim(0, p.shape[0])
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])

    with h5py.File('../../sample_3d.h5', 'r') as f:
        image = f['img'][:]
        target = f['target'][:]

    # plot_3d(image[..., 0], 1094-100)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # plot_3d(image[..., 0], ax, 1194, 0.1, (0.5, 0.5, 1), cut=(100, 150, 0))
    plot_3d(image[..., 0], ax, 1194, 0.3, (1, 1, 1))
    # plot_3d(image[..., 0], ax, 1000, 0.01, (0.5, 0.5, 0.8),)
    # plot_3d(image[..., 0], ax, 900, 0.01, (0.5, 0.5, 0.3),)

    plot_3d(target[..., 0], ax, 0.5, 0.9, (1, 0.5, 0.5))

    # plt.show()
    plt.savefig('3d.png')
