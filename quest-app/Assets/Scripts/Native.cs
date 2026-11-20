using System;
using System.Runtime.InteropServices;

// native/bbsim.c, sim.py's World on the headset. Only the sim thread (Sim.cs) calls these.
static class Native
{
    const string Lib = "bbsim";

    [DllImport(Lib)] public static extern IntPtr bb_create(byte[] xml, int len, byte[] err, int errlen);
    [DllImport(Lib)] public static extern void bb_free(IntPtr w);
    [DllImport(Lib)] public static extern void bb_set_params(IntPtr w, double massScale, double maxForce, int assist);
    [DllImport(Lib)] public static extern void bb_hand(IntPtr w, double[] pos, double[] quat);
    [DllImport(Lib)] public static extern void bb_set_hand(IntPtr w, double[] pos, double[] quat, int teleport);
    [DllImport(Lib)] public static extern void bb_reset_object(IntPtr w);
    [DllImport(Lib)] public static extern void bb_step(IntPtr w, double dt, double[] angles, double effort, int locked);
    [DllImport(Lib)] public static extern int bb_ngeom(IntPtr w);
    [DllImport(Lib)] public static extern void bb_geom(IntPtr w, int g, out int type, out int role, out int finger, double[] size);
    [DllImport(Lib)] public static extern int bb_mesh(IntPtr w, int g, float[] vert, int[] face, out int nface);
    [DllImport(Lib)] public static extern void bb_poses(IntPtr w, float[] pos, float[] mat);
    [DllImport(Lib)] public static extern void bb_readout(IntPtr w, double[] readout);
}
