using System.Collections.Generic;
using UnityEngine;

// Draws the sim's geoms under the table root, like web/contact.js does on the laptop. The sim is z up and right
// handed, Unity y up and left handed: the user sits on the sim's +y side looking towards -y (where the fingers point),
// so Unity (x, y, z) = (-x, z, -y) of the sim.
public class SimView
{
    // A finger pressing this hard shows fully in the effort colour (contact.js TINT_FULL_N)
    const float TintFullN = 10;
    public static readonly Color HandColor = Hex(0xd4d4d8), PressColor = Hex(0xc93ee6), ObjectColor = Hex(0x71717a),
        TableColor = Hex(0x151517);

    readonly Transform root;
    readonly Material material;
    readonly List<Transform> parts = new List<Transform>();
    readonly List<Renderer> fingers = new List<Renderer>();
    readonly List<int> fingerOf = new List<int>();
    int version = -1;
    float[] pos = new float[0], mat = new float[0];

    public SimView(Transform root, Material material)
    {
        this.root = root;
        this.material = material;
    }

    public static Color Hex(int rgb) => new Color((rgb >> 16 & 255) / 255f, (rgb >> 8 & 255) / 255f, (rgb & 255) / 255f);

    public static Vector3 ToUnity(double x, double y, double z) => new Vector3((float)-x, (float)z, (float)-y);

    public static void ToSim(Vector3 u, double[] p)
    {
        p[0] = -u.x;
        p[1] = -u.z;
        p[2] = u.y;
    }

    // A sim rotation matrix (row major at m[o]) as a Unity rotation: Unity's local y is the sim's local z and its local z
    // the sim's local -y, the same swap as for positions
    public static Quaternion ToUnity(float[] m, int o)
    {
        Vector3 up = ToUnity(m[o + 2], m[o + 5], m[o + 8]);
        Vector3 forward = -ToUnity(m[o + 1], m[o + 4], m[o + 7]);
        return Quaternion.LookRotation(forward, up);
    }

    public static Quaternion QuatToUnity(double[] h, int o)
    {
        double w = h[o], x = h[o + 1], y = h[o + 2], z = h[o + 3];
        var m = new float[]
        {
            (float)(1 - 2 * (y * y + z * z)), (float)(2 * (x * y - w * z)), (float)(2 * (x * z + w * y)),
            (float)(2 * (x * y + w * z)), (float)(1 - 2 * (x * x + z * z)), (float)(2 * (y * z - w * x)),
            (float)(2 * (x * z - w * y)), (float)(2 * (y * z + w * x)), (float)(1 - 2 * (x * x + y * y)),
        };
        return ToUnity(m, 0);
    }

    // A Unity rotation as a sim quaternion (w x y z), the inverse of ToUnity
    public static void QuatToSim(Quaternion q, double[] out4)
    {
        var c = new double[3, 3];
        Vector3[] axes = { q * Vector3.left, q * Vector3.back, q * Vector3.up };
        var p = new double[3];
        for (int k = 0; k < 3; k++)
        {
            ToSim(axes[k], p);
            for (int i = 0; i < 3; i++) c[i, k] = p[i];
        }
        double tr = c[0, 0] + c[1, 1] + c[2, 2], w, x, y, z;
        if (tr > 0)
        {
            double s = System.Math.Sqrt(tr + 1) * 2;
            w = s / 4; x = (c[2, 1] - c[1, 2]) / s; y = (c[0, 2] - c[2, 0]) / s; z = (c[1, 0] - c[0, 1]) / s;
        }
        else if (c[0, 0] > c[1, 1] && c[0, 0] > c[2, 2])
        {
            double s = System.Math.Sqrt(1 + c[0, 0] - c[1, 1] - c[2, 2]) * 2;
            w = (c[2, 1] - c[1, 2]) / s; x = s / 4; y = (c[0, 1] + c[1, 0]) / s; z = (c[0, 2] + c[2, 0]) / s;
        }
        else if (c[1, 1] > c[2, 2])
        {
            double s = System.Math.Sqrt(1 + c[1, 1] - c[0, 0] - c[2, 2]) * 2;
            w = (c[0, 2] - c[2, 0]) / s; x = (c[0, 1] + c[1, 0]) / s; y = s / 4; z = (c[1, 2] + c[2, 1]) / s;
        }
        else
        {
            double s = System.Math.Sqrt(1 + c[2, 2] - c[0, 0] - c[1, 1]) * 2;
            w = (c[1, 0] - c[0, 1]) / s; x = (c[0, 2] + c[2, 0]) / s; y = (c[1, 2] + c[2, 1]) / s; z = s / 4;
        }
        out4[0] = w; out4[1] = x; out4[2] = y; out4[3] = z;
    }

    Material Paint(Color c)
    {
        var m = new Material(material) { color = c };
        return m;
    }

    GameObject Primitive(PrimitiveType type)
    {
        var go = GameObject.CreatePrimitive(type);
        Object.Destroy(go.GetComponent<Collider>());
        return go;
    }

    // MuJoCo geom types: 0 plane, 2 sphere, 3 capsule, 4 ellipsoid, 5 cylinder, 6 box, 7 mesh
    Transform Make(Sim.Geom g)
    {
        var s = g.size;
        GameObject go;
        switch (g.type)
        {
            case 0:
                // Drawn as a slab that starts just in front of the user rather than the sim's 1 m square round the
                // origin; the plane collides everywhere anyway
                go = new GameObject("table");
                var slab = Primitive(PrimitiveType.Cube);
                slab.transform.SetParent(go.transform, false);
                slab.transform.localScale = new Vector3(1.0f, 0.02f, 0.7f);
                slab.transform.localPosition = new Vector3(0, -0.01f, 0.15f);
                slab.GetComponent<Renderer>().sharedMaterial = Paint(TableColor);
                return go.transform;
            case 2:
                go = Primitive(PrimitiveType.Sphere);
                go.transform.localScale = Vector3.one * (float)(2 * s[0]);
                break;
            case 3:
                go = new GameObject("capsule", typeof(MeshFilter), typeof(MeshRenderer));
                go.GetComponent<MeshFilter>().sharedMesh = Capsule((float)s[0], (float)s[1]);
                break;
            case 4:
                go = Primitive(PrimitiveType.Sphere);
                go.transform.localScale = new Vector3((float)(2 * s[0]), (float)(2 * s[2]), (float)(2 * s[1]));
                break;
            case 5:
                go = Primitive(PrimitiveType.Cylinder);
                go.transform.localScale = new Vector3((float)(2 * s[0]), (float)s[1], (float)(2 * s[0]));
                break;
            case 6:
                go = Primitive(PrimitiveType.Cube);
                go.transform.localScale = new Vector3((float)(2 * s[0]), (float)(2 * s[2]), (float)(2 * s[1]));
                break;
            case 7:
                go = new GameObject("mesh", typeof(MeshFilter), typeof(MeshRenderer));
                go.GetComponent<MeshFilter>().sharedMesh = FlatMesh(g.vert, g.face);
                break;
            default:
                return null;
        }
        go.GetComponent<Renderer>().sharedMaterial = Paint(g.role == 1 ? ObjectColor : HandColor);
        return go.transform;
    }

    // Capsule of radius r and half length h along local y, MuJoCo's capsule after the axis swap
    static Mesh Capsule(float r, float h)
    {
        const int seg = 16, ring = 6;
        var v = new List<Vector3>();
        var n = new List<Vector3>();
        var t = new List<int>();
        int rows = 2 * ring + 2;
        for (int i = 0; i < rows; i++)
        {
            bool top = i <= ring;
            float phi = Mathf.PI / 2 * (top ? (float)i / ring : 1 + (float)(i - ring - 1) / ring);
            for (int j = 0; j <= seg; j++)
            {
                float th = 2 * Mathf.PI * j / seg;
                var dir = new Vector3(Mathf.Sin(phi) * Mathf.Cos(th), Mathf.Cos(phi), Mathf.Sin(phi) * Mathf.Sin(th));
                n.Add(dir);
                v.Add(dir * r + Vector3.up * (top ? h : -h));
            }
        }
        for (int i = 0; i < rows - 1; i++)
            for (int j = 0; j < seg; j++)
            {
                int a = i * (seg + 1) + j, b = a + seg + 1;
                t.AddRange(new[] { a, a + 1, b, a + 1, b + 1, b });
            }
        var mesh = new Mesh();
        mesh.SetVertices(v);
        mesh.SetNormals(n);
        mesh.SetTriangles(t, 0);
        return mesh;
    }

    // MuJoCo mesh in the geom's frame, one vertex per face corner for flat shading; the axis swap flips handedness,
    // so the winding is reversed
    static Mesh FlatMesh(float[] vert, int[] face)
    {
        var v = new List<Vector3>();
        var t = new List<int>();
        for (int f = 0; f < face.Length; f += 3)
            for (int k = 2; k >= 0; k--)
            {
                int i = 3 * face[f + k];
                t.Add(v.Count);
                v.Add(ToUnity(vert[i], vert[i + 1], vert[i + 2]));
            }
        var mesh = new Mesh();
        mesh.SetVertices(v);
        mesh.SetTriangles(t, 0);
        mesh.RecalculateNormals();
        return mesh;
    }

    void Rebuild(Sim.Geom[] geoms)
    {
        foreach (var p in parts) if (p != null) Object.Destroy(p.gameObject);
        parts.Clear();
        fingers.Clear();
        fingerOf.Clear();
        foreach (var g in geoms)
        {
            var part = Make(g);
            if (part != null)
            {
                part.SetParent(root, false);
                if (g.role == 2)
                {
                    fingers.Add(part.GetComponent<Renderer>());
                    fingerOf.Add(g.finger);
                }
            }
            parts.Add(part);
        }
    }

    // Moves every geom to the sim's newest poses and tints pressing fingers; readout and hand come back for the app
    public bool Update(Sim sim, double[] readout, double[] hand)
    {
        if (!sim.Read(ref pos, ref mat, readout, hand)) return false;
        if (sim.Version != version)
        {
            version = sim.Version;
            Rebuild(sim.Geoms);
        }
        if (pos.Length != 3 * parts.Count) return false;
        for (int g = 0; g < parts.Count; g++)
        {
            var part = parts[g];
            if (part == null || part.name == "table") continue;
            part.localPosition = ToUnity(pos[3 * g], pos[3 * g + 1], pos[3 * g + 2]);
            part.localRotation = ToUnity(mat, 9 * g);
        }
        for (int i = 0; i < fingers.Count; i++)
        {
            float force = fingerOf[i] >= 0 ? (float)readout[1 + fingerOf[i]] : 0;
            fingers[i].sharedMaterial.color = Color.Lerp(HandColor, PressColor, Mathf.Clamp01(force / TintFullN));
        }
        return true;
    }
}
