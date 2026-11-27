using System.Collections.Generic;
using TMPro;
using UnityEngine;

// The opening in place of Unity's splash: the web app's logo (the gradient ring, then the pulse line through it)
// drawing itself in, the wordmark under it, then what it still waits for (the sim, the laptop) until both are up or
// the laptop isn't found. Fades out and leaves the room to the app.
public class Intro
{
    const float RingS = 0.8f, LineFrom = 0.5f, LineS = 0.8f, WordFrom = 1.1f, WordS = 0.5f, StatusFrom = 2.0f;
    // Shown at least this long; without the laptop it gives up after LaptopWaitS
    const float MinS = 2.2f, LaptopWaitS = 6, FadeS = 0.4f;
    const float Size = 0.32f, Distance = 1.1f;
    // The Quest's own loading view takes about this long to fade once the app draws, so the logo waits for it
    const float StartS = 0.8f;
    // The favicon's drawing in its 32 unit box: ring round (16, 16) with radius 12, the pulse line's points
    static readonly Vector2[] Pulse =
    {
        new Vector2(7, 16), new Vector2(11, 16), new Vector2(13, 10), new Vector2(16, 22), new Vector2(19, 8),
        new Vector2(21, 18), new Vector2(22.5f, 16), new Vector2(25, 16)
    };

    readonly Transform root;
    readonly MeshFilter ring, line;
    readonly TextMeshPro word, status;
    float t, wait, fadeAt = -1;
    bool placed;

    public bool Done { get; private set; }

    public Intro(Material material)
    {
        root = new GameObject("intro").transform;
        ring = Ui.WorldShape(root, material, Vector3.zero, Vector2.one, 0, Color.white, 5);
        line = Ui.WorldShape(root, material, Vector3.zero, Vector2.one, 0, Color.white, 6);
        word = Ui.WorldLabel(root, new Vector3(0, -Size * 0.72f, 0), "BETTER BIONICS", 0.032f, Ui.Text, false, true,
            TextAlignmentOptions.Center, new Vector2(0.6f, 0.05f));
        word.characterSpacing = 18;
        word.enableVertexGradient = true;
        word.colorGradient = new VertexGradient(Ui.Orange, Ui.Magenta, Ui.Orange, Ui.Magenta);
        status = Ui.WorldLabel(root, new Vector3(0, -Size * 0.72f - 0.06f, 0), "", 0.018f, Ui.Muted, false, false,
            TextAlignmentOptions.Center, new Vector2(0.6f, 0.04f));
        SetAlpha(0);
    }

    // Logo units (0 to 32, y down) to metres round the logo's centre
    static Vector3 At(Vector2 u) => new Vector3((u.x - 16) / 32 * Size, -(u.y - 16) / 32 * Size, 0);

    static readonly Vector3[] PulseAt = System.Array.ConvertAll(Pulse, At);

    public void Update(Transform head, bool simReady, bool linked)
    {
        if (Done) return;
        wait += Time.deltaTime;
        if (wait < StartS) return;
        t += Time.deltaTime;
        // In front of the eyes once the head is tracked, following it lazily so it never sits off to the side
        var forward = Vector3.ProjectOnPlane(head.forward, Vector3.up).normalized;
        if (forward.sqrMagnitude > 0.5f && head.position.y > 0.3f)
        {
            var goal = head.position + forward * Distance;
            float k = placed ? 1 - Mathf.Exp(-Time.deltaTime / 0.4f) : 1;
            root.position = Vector3.Lerp(root.position, goal, k);
            root.rotation = Quaternion.Slerp(root.rotation, Quaternion.LookRotation(forward, Vector3.up), k);
            placed = true;
        }

        Replace(ring, Arc(Ease(Mathf.Clamp01(t / RingS)), 12f / 32 * Size, 3f / 32 * Size, Size));
        Replace(line, Polyline(PulseAt, Ease(Mathf.Clamp01((t - LineFrom) / LineS)), 2.2f / 32 * Size, Ui.Text));
        float wordA = Mathf.Clamp01((t - WordFrom) / WordS);
        if (fadeAt < 0 && t > MinS && simReady && (linked || t > LaptopWaitS)) fadeAt = t;
        status.text = t < StatusFrom ? "" : !simReady ? "Starting the hand"
            : linked ? "Laptop connected" : t > LaptopWaitS ? "Laptop not found, starting without it"
            : "Looking for the laptop over USB";
        float fade = fadeAt < 0 ? 1 : 1 - Mathf.Clamp01((t - fadeAt) / FadeS);
        SetAlpha(fade, wordA);
        if (fadeAt >= 0 && t - fadeAt >= FadeS)
        {
            Done = true;
            Object.Destroy(ring.sharedMesh);
            Object.Destroy(line.sharedMesh);
            Object.Destroy(root.gameObject);
        }
    }

    public static void Replace(MeshFilter mf, Mesh mesh)
    {
        if (mf.sharedMesh != null) Object.Destroy(mf.sharedMesh);
        mf.sharedMesh = mesh;
    }

    public static float Ease(float x) => 1 - (1 - x) * (1 - x) * (1 - x);

    void SetAlpha(float a, float wordA = 0)
    {
        foreach (var mf in new[] { ring, line })
        {
            var m = mf.sharedMesh;
            if (m == null || m.vertexCount == 0) continue;
            var c = m.colors;
            for (int i = 0; i < c.Length; i++) c[i].a = a;
            m.colors = c;
        }
        word.alpha = a * wordA;
        status.alpha = a;
    }

    // A ring of radius r from the top, clockwise, done of the way round; coloured across x like the site's gradient,
    // over span (m). Also the finish card's (Finish.cs)
    public static Mesh Arc(float done, float r, float width, float span)
    {
        var v = new List<Vector3>();
        int n = Mathf.Max(2, Mathf.CeilToInt(96 * done) + 1);
        for (int i = 0; i < n; i++)
        {
            float a = Mathf.PI / 2 - 2 * Mathf.PI * done * i / (n - 1);
            var d = new Vector3(Mathf.Cos(a), Mathf.Sin(a), 0);
            v.Add(d * (r - width / 2));
            v.Add(d * (r + width / 2));
        }
        return Strip(v, span);
    }

    // A line through points (m) drawn from the first, done of its length: a quad per segment and a round joint at
    // each point reached, like the favicon's round line joins. Also the finish card's check and cross
    public static Mesh Polyline(Vector3[] points, float done, float width, Color color)
    {
        float total = 0;
        for (int i = 1; i < points.Length; i++) total += Vector3.Distance(points[i - 1], points[i]);
        float left = total * done;
        var v = new List<Vector3>();
        var tris = new List<int>();
        for (int i = 1; i < points.Length && left > 0; i++)
        {
            var a = points[i - 1];
            float len = Vector3.Distance(a, points[i]);
            var b = Vector3.Lerp(a, points[i], Mathf.Min(1, left / len));
            left -= len;
            var dir = (b - a).normalized;
            var side = new Vector3(-dir.y, dir.x, 0) * (width / 2);
            int s = v.Count;
            v.AddRange(new[] { a - side, a + side, b - side, b + side });
            tris.AddRange(new[] { s, s + 1, s + 2, s + 1, s + 3, s + 2 });
            Disc(v, tris, a, width / 2);
            Disc(v, tris, b, width / 2);
        }
        var mesh = new Mesh();
        mesh.SetVertices(v);
        // Both faces
        int n = tris.Count;
        for (int i = 0; i < n; i += 3) tris.AddRange(new[] { tris[i], tris[i + 2], tris[i + 1] });
        mesh.SetTriangles(tris, 0);
        var colors = new Color[v.Count];
        for (int i = 0; i < colors.Length; i++) colors[i] = color;
        mesh.colors = colors;
        return mesh;
    }

    static void Disc(List<Vector3> v, List<int> tris, Vector3 c, float r)
    {
        const int sides = 12;
        int s = v.Count;
        v.Add(c);
        for (int i = 0; i < sides; i++)
        {
            float a = 2 * Mathf.PI * i / sides;
            v.Add(c + new Vector3(Mathf.Cos(a), Mathf.Sin(a), 0) * r);
            tris.AddRange(new[] { s, s + 1 + (i + 1) % sides, s + 1 + i });
        }
    }

    // A strip of quads as a mesh, two vertices per step (the ring). Both faces, coloured by x
    static Mesh Strip(List<Vector3> v, float span)
    {
        var mesh = new Mesh();
        if (v.Count < 4) return mesh;
        var tris = new List<int>();
        for (int i = 0; i + 3 < v.Count; i += 2)
            tris.AddRange(new[] { i, i + 1, i + 2, i + 1, i + 3, i + 2, i, i + 2, i + 1, i + 1, i + 2, i + 3 });
        mesh.SetVertices(v);
        mesh.SetTriangles(tris, 0);
        var colors = new Color[v.Count];
        for (int i = 0; i < colors.Length; i++)
            colors[i] = Ui.GradientAt(Mathf.InverseLerp(-span / 2, span / 2, v[i].x));
        mesh.colors = colors;
        return mesh;
    }
}
