using System;
using System.Runtime.InteropServices;

// native/bbsim.c, sim.py's World on the headset with two hands. Hand 0 is the prosthetic one, 1 the free one. Only
// the sim thread (Sim.cs) calls the world's functions.
static class Native
{
    const string Lib = "bbsim";
    public const int ReadoutSize = 17;

    [DllImport(Lib)] public static extern IntPtr bb_create(byte[] xml, int len, byte[] err, int errlen);
    [DllImport(Lib)] public static extern void bb_free(IntPtr w);
    [DllImport(Lib)] public static extern void bb_set_params(IntPtr w, double massScale, double maxForce, int assist);
    [DllImport(Lib)] public static extern void bb_set_input(IntPtr w, int h, double[] angles, double effort, int locked, int grab);
    [DllImport(Lib)] public static extern void bb_set_hand(IntPtr w, int h, double[] pos, double[] quat, int teleport);
    [DllImport(Lib)] public static extern void bb_place_object(IntPtr w, int o, double[] pos, double[] quat, int parked);
    [DllImport(Lib)] public static extern int bb_nobjects(IntPtr w);
    [DllImport(Lib)] public static extern void bb_set_floor(IntPtr w, double height);
    [DllImport(Lib)] public static extern void bb_step(IntPtr w, double dt);
    [DllImport(Lib)] public static extern int bb_ngeom(IntPtr w);
    [DllImport(Lib)] public static extern void bb_geom(IntPtr w, int g, out int type, out int role, out int hand, out int finger, out int obj, double[] size);
    [DllImport(Lib)] public static extern int bb_mesh(IntPtr w, int g, float[] vert, int[] face, out int nface);
    [DllImport(Lib)] public static extern void bb_poses(IntPtr w, float[] pos, float[] mat);
    [DllImport(Lib)] public static extern void bb_object_pose(IntPtr w, int o, double[] pos, double[] quat);
    [DllImport(Lib)] public static extern void bb_readout(IntPtr w, double[] readout);

    // The ghost hand (a cue's target pose): kinematics only, main thread
    [DllImport(Lib)] public static extern IntPtr bb_fk_create(byte[] xml, int len, byte[] err, int errlen);
    [DllImport(Lib)] public static extern void bb_fk_free(IntPtr fk);
    [DllImport(Lib)] public static extern int bb_fk_ngeom(IntPtr fk);
    [DllImport(Lib)] public static extern void bb_fk_geom(IntPtr fk, int g, out int type, out int finger, double[] size);
    [DllImport(Lib)] public static extern int bb_fk_mesh(IntPtr fk, int g, float[] vert, int[] face, out int nface);
    [DllImport(Lib)] public static extern void bb_fk(IntPtr fk, double[] angles, float[] pos, float[] mat);
    [DllImport(Lib)] public static extern void bb_fk_points(IntPtr fk, double[] angles, float[] points);
}
