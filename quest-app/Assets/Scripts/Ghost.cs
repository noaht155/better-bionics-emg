using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

// A hand on its own, posed by angles, for showing what to do: the recording cue's target. Built from the same hand
// model as the sim (export_model.ghost_xml) with MuJoCo's kinematics, so the pose is exactly what the angles mean.
public class Ghost
{
    readonly Transform root;
    readonly List<Transform> parts = new List<Transform>();
    readonly double[] angles = new double[20];
    IntPtr fk;
    float[] pos, mat;

    public Transform Root => root;

    public Ghost(string xml, Material material, Color color)
    {
        root = new GameObject("ghost hand").transform;
        var bytes = Encoding.UTF8.GetBytes(xml);
        var err = new byte[500];
        fk = Native.bb_fk_create(bytes, bytes.Length, err, err.Length);
        if (fk == IntPtr.Zero)
        {
            Debug.Log("ghost hand: " + Encoding.UTF8.GetString(err).TrimEnd('\0'));
            return;
        }
        int n = Native.bb_fk_ngeom(fk);
        pos = new float[3 * n];
        mat = new float[9 * n];
        var paint = SimView.Paint(material, color, 0.6f);
        var size = new double[3];
        for (int g = 0; g < n; g++)
        {
            Native.bb_fk_geom(fk, g, out int type, out _, size);
            var geom = new Sim.Geom { type = type, role = 2, size = (double[])size.Clone() };
            int nv = Native.bb_fk_mesh(fk, g, null, null, out int nf);
            if (nv > 0)
            {
                geom.vert = new float[3 * nv];
                geom.face = new int[3 * nf];
                Native.bb_fk_mesh(fk, g, geom.vert, geom.face, out nf);
            }
            var part = SimView.MakePart(geom, paint);
            part.SetParent(root, false);
            parts.Add(part);
        }
    }

    public void Pose(float[] degrees)
    {
        if (fk == IntPtr.Zero) return;
        for (int j = 0; j < 20; j++) angles[j] = degrees[j];
        Native.bb_fk(fk, angles, pos, mat);
        for (int g = 0; g < parts.Count; g++)
        {
            parts[g].localPosition = SimView.ToUnity(pos[3 * g], pos[3 * g + 1], pos[3 * g + 2]);
            parts[g].localRotation = SimView.ToUnity(mat, 9 * g);
        }
    }

    public void Destroy()
    {
        if (fk != IntPtr.Zero) Native.bb_fk_free(fk);
        fk = IntPtr.Zero;
        UnityEngine.Object.Destroy(root.gameObject);
    }
}
