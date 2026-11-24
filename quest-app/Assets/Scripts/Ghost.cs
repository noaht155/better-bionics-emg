using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

// A hand on its own, posed by angles, for showing what to do: the recording cue's target. Built from the same hand
// model as the sim (export_model.ghost_xml) with MuJoCo's kinematics, so the pose is exactly what the angles mean.
public class Ghost
{
    readonly Transform root, wrist, forearm;
    readonly List<Transform> parts = new List<Transform>();
    readonly double[] angles = new double[20];
    IntPtr fk;
    float[] pos, mat;

    public Transform Root => root;

    public Ghost(string xml, Material material, Color color)
    {
        root = new GameObject("ghost hand").transform;
        // The hand turns at the wrist for the wrist cues; the forearm stays and only shows for them
        wrist = new GameObject("wrist").transform;
        wrist.SetParent(root, false);
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
        forearm = SimView.MakePart(new Sim.Geom { type = 3, size = new double[] { 0.024, 0.11, 0 } }, paint);
        forearm.SetParent(root, false);
        // MuJoCo's capsule lies along sim z, Unity y; this one goes back from the wrist along sim -y, Unity +z
        forearm.localPosition = SimView.ToUnity(0, -0.13, 0);
        forearm.localRotation = Quaternion.Euler(90, 0, 0);
        forearm.gameObject.SetActive(false);
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
            part.SetParent(wrist, false);
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

    // Bends the hand at the wrist (degrees, towards the palm positive: about the palm's x axis, which turns the
    // fingers towards the palm side), and shows the forearm while a wrist cue runs
    public void SetWrist(float flex, bool showArm)
    {
        float a = flex * Mathf.Deg2Rad / 2;
        wristQuat[0] = Mathf.Cos(a);
        wristQuat[1] = Mathf.Sin(a);
        wrist.localRotation = SimView.QuatToUnity(wristQuat, 0);
        forearm.gameObject.SetActive(showArm);
    }

    readonly double[] wristQuat = new double[4];

    public void Destroy()
    {
        if (fk != IntPtr.Zero) Native.bb_fk_free(fk);
        fk = IntPtr.Zero;
        UnityEngine.Object.Destroy(root.gameObject);
    }
}
