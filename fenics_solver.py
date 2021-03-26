# -*- encoding: utf-8 -*-
"""
Desc      :   solver for layout-generator equation.
"""
# File    :   fenics_solver.py
# Time    :   2020/03/29 15:16:48
# Author  :   Zweien
# Contact :   278954153@qq.com


import fenics as fs
import matplotlib.pyplot as plt
import numpy as np
import random

# fs.set_log_level(40)  # ERROR = 40
TOL = 1e-14


def layout2matrix(ndim, nx, unit_per_row, powers, layout_pos_list):
    """将 layout 位置 list 转换为矩阵
    """
    assert ndim in [2, 3]
    F = np.zeros((nx + 1,) * ndim)
    if ndim == 3:
        for i, pos in enumerate(layout_pos_list):
            z = pos // (unit_per_row ** 2)
            y = (pos % (unit_per_row ** 2)) // unit_per_row
            x = pos % unit_per_row
            size = int((nx + 1) / unit_per_row)
            x_slice = slice(size * x, size * (x + 1))
            y_slice = slice(size * y, size * (y + 1))
            z_slice = slice(size * z, size * (z + 1))
            F[x_slice, y_slice, z_slice] = powers[i]
    else:
        for i, pos in enumerate(layout_pos_list):
            x, y = pos % unit_per_row, pos // unit_per_row
            size = int((nx + 1) / unit_per_row)
            x_slice = slice(size * x, size * (x + 1))
            y_slice = slice(size * y, size * (y + 1))
            F[y_slice, x_slice] = powers[i]
    return F


class Source(fs.UserExpression):
    """热源布局"""

    def __init__(self, layouts, length, length_unit, powers):
        """

        Arguments:
            layouts {list or int} -- 组件摆放位置
            length {float} -- 布局板尺寸
            length_unit {float} -- 组件尺寸
        """
        super().__init__(self)
        self.layout_list = layouts if isinstance(layouts, list) else [layouts]
        self.length = length
        self.length_unit = length_unit
        self.n = length / length_unit  # unit_per_row
        self.powers = powers

    def eval(self, value, x):
        value[0] = self.get_source(x)

    def get_source(self, x):
        for _, (l, power) in enumerate(zip(self.layout_list, self.powers)):
            lx, ly = l % self.n, l // self.n
            if (
                    self.length_unit * lx <= x[0] <= self.length_unit * (lx + 1)
            ) and (
                    self.length_unit * ly <= x[1] <= self.length_unit * (ly + 1)
            ):
                return power

        return 0

    def value_shape(self):
        return ()


class SourceF(fs.UserExpression):
    def __init__(self, F, length):
        """

        Args:
            F (ndarray): 热源矩阵，2d or 3d
            length (float): 板边长
        """
        super().__init__(self)
        self.F = F
        self.ndim = F.ndim
        self.length = length

    def eval(self, value, x):
        value[0] = self.get_source(x)

    def get_source(self, x):
        """由预生成的 F 获取热源函数值 f(x).

        Args:
            x : 坐标

        Returns:
            float: 热源函数值 f(x)
        """
        assert self.ndim in [2, 3]
        n = self.F.shape[0]
        if self.ndim == 2:
            xx = int(x[0] / self.length * (n - 1))
            yy = int(x[1] / self.length * (n - 1))
            return self.F[yy, xx]
        xx = int(x[0] / self.length * (n - 1))
        yy = int(x[1] / self.length * (n - 1))
        zz = int(x[2] / self.length * (n - 1))
        return self.F[zz, yy, xx]

    def value_shape(self):
        return ()


class LineBoundary:
    """线段边界

    Args:
        line (list): 表示边界的线段，格式为 [[起点x, 起点y], [终点x, 终点y]]
    """

    def __init__(self, line):

        self.line = line
        assert len(line) == 2, "线段包含两个点"
        assert len(line[0]) == 2 and len(line[1]) == 2, "二维点"

    def get_boundary(self):
        """构造 fenics 所需 bc 函数

        Returns:
            function: fenics 所需 bc
        """

        def boundary(x, on_boundary):
            if on_boundary:
                (lx, ly), (rx, ry) = self.line
                if (lx - TOL <= x[0] <= rx + TOL) and (
                        ly - TOL <= x[1] <= ry + TOL
                ):
                    return True
            return False

        return boundary


class RecBoundary:
    """线段边界

    Args:
        rec (list): 表示边界的矩形，格式为 [[起点x, 起点y, 起点z], [终点x, 终点y, 终点z]]
    """

    def __init__(self, rec):
        self.rec = rec
        assert len(rec) == 2, "线段必须包含两个点"
        assert len(rec[0]) == 3 and len(rec[1]) == 3, "必须为三维点"

    def get_boundary(self):
        """构造 fenics 所需 bc 函数

        Returns:
            function: fenics 所需 bc
        """

        def boundary(x, on_boundary):
            if on_boundary:
                (lx, ly, lz), (rx, ry, rz) = self.rec
                if (
                        (lx - TOL <= x[0] <= rx + TOL)
                        and (ly - TOL <= x[1] <= ry + TOL)
                        and (lz - TOL <= x[2] <= rz + TOL)
                ):
                    return True
            return False

        return boundary


def solver(f, u_D, bc_funs, ndim, length, nx, ny, nz=None, degree=2):
    """Fenics 求解器

    Args:
        f (Expression): [description]
        u_D (Expression): [description]
        bc_funs (List[Callable]): [description]
        ndim (int): [description]
        length (float): [description]
        nx (int): [description]
        ny (int): [description]
        nz (int, optional): [description]. Defaults to None.
        degree (int, optional): [description]. Defaults to 1.

    Returns:
        Function: 解 u
    """

    mesh = get_mesh(length, nx, ny, nz)
    # mesh = refine_mesh(mesh, length)

    V = fs.FunctionSpace(mesh, "P", degree)
    bcs = [fs.DirichletBC(V, u_D, bc) for bc in bc_funs]
    u = fs.TrialFunction(V)
    v = fs.TestFunction(V)
    FF = fs.dot(fs.grad(u), fs.grad(v)) * fs.dx - f * v * fs.dx
    a = fs.lhs(FF)
    L = fs.rhs(FF)
    u = fs.Function(V)
    fs.solve(a == L, u, bcs)
    return u, mesh


def get_mesh(length, nx, ny, nz=None):
    """获得 mesh

    """
    if nz is None:
        mesh = fs.RectangleMesh(
            fs.Point(0.0, 0.0), fs.Point(length, length), nx, ny
        )
    else:
        mesh = fs.BoxMesh(
            fs.Point(0.0, 0.0, 0.0),
            fs.Point(length, length, length),
            nx,
            ny,
            nz,
        )
    return mesh


def refine_mesh(mesh, length):
    def y_refine(y, length):
        y = np.power(y, 2) / length
        return y

    def x_refine(x, length):
        x = (4.0 / (length * length)) * (np.power(x - length / 2, 3)) + length / 2
        return x

    def mesh_refine(x, y):
        return [x_refine(x, length), y_refine(y, length)]

    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]

    x_bar, y_bar = mesh_refine(x, y)
    xy_bar_coor = np.array([x_bar, y_bar]).transpose()
    mesh.coordinates()[:] = xy_bar_coor
    return mesh


def get_mesh_grid(length, nx, ny, nz=None):
    """获取网格节点坐标
    """
    mesh = get_mesh(length, nx, ny, nz)
    # mesh = refine_mesh(mesh, length)
    if nz is None:
        xs = mesh.coordinates()[:, 0].reshape(nx + 1, nx + 1)
        ys = mesh.coordinates()[:, 1].reshape(nx + 1, ny + 1)
        return xs, ys, None
    xs = mesh.coordinates()[:, 0].reshape(nx + 1, ny + 1, nz + 1)
    ys = mesh.coordinates()[:, 1].reshape(nx + 1, ny + 1, nz + 1)
    zs = mesh.coordinates()[:, 2].reshape(nx + 1, ny + 1, nz + 1)
    return xs, ys, zs


def run_solver(
        ndim,
        length,
        length_unit,
        bcs,
        layout_list,
        u0,
        powers,
        nx,
        coordinates=False,
        is_plot=False,
        F=None,
        vtk=False,
):
    """求解器主函数.

    Args:
        ndim (int): 2 or 3, 问题维数
        length (float): board length
        length_unit (float): unit length
        bcs (list): bcs
        layout_list (list): unit 位置
        u0 (float): Dirichlet bc 上的值
        powers (list): 功率 list
        nx (int): x 方向上的单元数
        coordinates (bool, optional): 是否返回坐标矩阵. Defaults to False.
        is_plot (bool, optional): 是否画图. Defaults to False.
        F (ndarray, optional): 热源布局矩阵 F. Defaults to None.
        vtk (bool): 是否输出 vtk 文件.

    Returns:
        tuple: U, xs, ys, zs
    """
    ny = nx
    nz = nx if ndim == 3 else None
    u_D = fs.Constant(u0)
    if len(bcs) > 0 and bcs[0] != []:
        if ndim == 2:
            bc_funs = [LineBoundary(line).get_boundary() for line in bcs]
        else:
            bc_funs = [RecBoundary(rec).get_boundary() for rec in bcs]
    else:
        bc_funs = [lambda x, on_boundary: on_boundary]  # 边界都为 Dirichlet

    if F is None:
        f = Source(layout_list, length, length_unit, powers)
    else:
        f = SourceF(F, length)
    u, mesh = solver(f, u_D, bc_funs, ndim, length, nx, ny, nz)
    if is_plot:
        fs.plot(u)
        # plt.savefig('demo_refine.png')
        # plt.show()
        # fs.plot(mesh, title='mesh')
        # plt.savefig('demo_refine_mesh.png')
        plt.show()
    if vtk:
        vtkfile = fs.File("solution.pvd")
        vtkfile << u
    if ndim == 2:
        U = u.compute_vertex_values().reshape(nx + 1, nx + 1)
    else:
        U = u.compute_vertex_values().reshape(nx + 1, nx + 1, nx + 1)
    if coordinates:
        xs, ys, zs = get_mesh_grid(length, nx, ny, nz)
    else:
        xs, ys, zs = None, None, None
    return U, xs, ys, zs


def run_solver_time_dependent(
        ndim, length, length_unit, bcs, layout_list, u0, powers, nx,
        coordinates=False, is_plot=False, F=None, vtk=False,
):
    """求解器主函数.

    Args:
        ndim (int): 2 or 3, 问题维数
        length (float): board length
        length_unit (float): unit length
        bcs (list): bcs
        layout_list (list): unit 位置
        u0 (float): Dirichlet bc 上的值
        powers (list): 功率 list
        nx (int): x 方向上的单元数
        coordinates (bool, optional): 是否返回坐标矩阵. Defaults to False.
        is_plot (bool, optional): 是否画图. Defaults to False.
        F (ndarray, optional): 热源布局矩阵 F. Defaults to None.
        vtk (bool): 是否输出 vtk 文件.

    Returns:
        tuple: U, xs, ys, zs
    """
    T = 100.0  # final time
    num_steps = 200  # number of time steps
    dt = T / num_steps  # time step size
    degree = 1  # degree of function space
    alpha = 50  # parameter alpha

    # Create mesh and define function space
    ny = nx
    nz = nx if ndim == 3 else None
    mesh = get_mesh(length, nx, ny, nz)
    V = fs.FunctionSpace(mesh, "P", degree)

    # Define boundary condition
    if len(bcs) > 0 and bcs[0] != []:
        if ndim == 2:
            bc_funs = [LineBoundary(line).get_boundary() for line in bcs]
        else:
            bc_funs = [RecBoundary(rec).get_boundary() for rec in bcs]
    else:
        bc_funs = [lambda x, on_boundary: on_boundary]  # 边界都为 Dirichlet
    # u_D = fs.Constant(u0)
    u_D = fs.Expression('10 * sin(t / alpha) + u0', degree=2, alpha=alpha, t=0, u0=u0)
    bcs = [fs.DirichletBC(V, u_D, bc) for bc in bc_funs]

    # Define initial value
    u_n = fs.interpolate(u_D, V)

    # Define viriational problem
    # 刚的密度和比热容7.9g/cm3和0.46kJ/Kg*C
    p, c = 7.9, 0.46
    pc = p * c * 1000
    u = fs.TrialFunction(V)
    v = fs.TestFunction(V)
    if F is None:
        f = Source(layout_list, length, length_unit, powers)
    else:
        f = SourceF(F, length)

    FF = dt * fs.dot(fs.grad(u), fs.grad(v)) * fs.dx + pc * u * v * fs.dx - (dt * f + pc * u_n) * v * fs.dx
    a, L = fs.lhs(FF), fs.rhs(FF)

    # Time-stepping
    u = fs.Function(V)
    t = 0
    if vtk:
        vtkfile = fs.File("poisson/solution.pvd")
    for n in range(num_steps):

        # Update current time
        t += dt

        # Update boundary condition
        u_D.t = t
        bcs = [fs.DirichletBC(V, u_D, bc) for bc in bc_funs]

        # Compute solution
        fs.solve(a == L, u, bcs)

        # Save to file and plot solution
        if vtk:
            vtkfile << u
        if is_plot:
            fs.plot(u)
            plt.show()
            # plt.colorbar()

        # Update previous solution
        u_n.assign(u)

    # if ndim == 2:
    #     U = u.compute_vertex_values().reshape(nx + 1, nx + 1)
    # else:
    #     U = u.compute_vertex_values().reshape(nx + 1, nx + 1, nx + 1)
    # if coordinates:
    #     xs, ys, zs = get_mesh_grid(length, nx, ny, nz)
    # else:
    #     xs, ys, zs = None, None, None
    return u  # U, xs, ys, zs


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 10000, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    # out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out * 10000


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == "__main__":
    # bcs = [[[0, 0.045], [0, 0.055]], [[0.1, 0.045], [0.1, 0.055]], [[0.045, 0], [0.055, 0]], [[0.045, 0.1], [0.055, 0.1]]]
    bcs = [[[0.0495, 0], [0.0505, 0]]]  # 小孔位置
    # bcs = [[[0, 0], [0.1, 0]]]
    layout_list = [0, 9, 11, 18, 22, 27, 33, 36, 44, 45, 54, 55, 63, 66, 72, 77, 81, 88, 90, 98]
    powers = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,
              10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
    # powers = [x * 2 for x in powers]
    nx = 199
    is_plot = True
    is_vtk = True
    coordinates = True
    F = layout2matrix(ndim=2, nx=nx, unit_per_row=10, powers=powers, layout_pos_list=layout_list)
    # run_solver_time_dependent(
    #     ndim=2, length=0.1, length_unit=0.01, bcs=bcs, layout_list=layout_list, u0=298,
    #     powers=powers, nx=nx, coordinates=False, is_plot=is_plot, F=None, vtk=is_vtk,
    # )

    #F = sp_noise(F, prob=0.02)
    # 添加高斯噪声，均值为0，方差为0.001
    F = gasuss_noise(F, mean=0, var=0.01)
    # G = np.random.randint(0, 2, (200, 200))
    # F = ((F / 10000) * G) * 10000

    U, xs, ys, zs = run_solver(ndim=2, length=0.1, length_unit=0.01, bcs=bcs, layout_list=layout_list, u0=298,
                               powers=powers, nx=nx, coordinates=coordinates, is_plot=is_plot, F=F, vtk=is_vtk)
    import scipy.io as sio

    sio.savemat('Example_sp_noise.mat', {"F": F, 'U': U, 'xs': xs, 'ys': ys})
