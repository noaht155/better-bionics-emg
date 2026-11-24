using UnityEngine;

// The room around the sim: a round room with soft walls, a floor that is lighter where the user sits, a ceiling light
// over the desk, and the desk, sized like the sim's desk (export_model.py) with its top at the desk root and legs
// down to the real floor. Dark, but lit like a room rather than a void.
public class Room
{
    const float Radius = 4, Height = 3;
    readonly Transform desk, lamp;
    readonly GameObject surroundings;
    readonly Transform[] legs = new Transform[4];
    readonly Vector2 half;
    readonly float centreZ;

    static readonly Color Skirting = SimView.Hex(0x1d1c21), WallLow = SimView.Hex(0x28272d),
        WallEye = SimView.Hex(0x323138), WallTop = SimView.Hex(0x1a1a1e), Ceiling = SimView.Hex(0x17171b),
        LampFace = SimView.Hex(0xe9e4da), LampGlow = SimView.Hex(0x45434b), FloorCentre = SimView.Hex(0x2e2d33),
        FloorEdge = SimView.Hex(0x1c1b20);

    public Room(Transform deskRoot, Material material, Material unlit, Vector2 deskHalf, float deskCentreY,
        float thickness)
    {
        half = deskHalf;
        centreZ = -deskCentreY;
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Exponential;
        RenderSettings.fogDensity = 0.05f;
        RenderSettings.fogColor = WallLow;

        surroundings = new GameObject("surroundings");
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "floor";
        Object.Destroy(floor.GetComponent<Collider>());
        floor.transform.SetParent(surroundings.transform, false);
        // The plane is 10 m, 2 Radius across here so it ends under the wall
        floor.transform.localScale = new Vector3(2 * Radius / 10, 1, 2 * Radius / 10);
        var fm = SimView.Paint(material, Color.white, 0.15f);
        fm.mainTexture = FloorTexture();
        floor.GetComponent<Renderer>().sharedMaterial = fm;

        // Walls and ceiling carry their colours in their vertices and aren't lit, like the menu. The unlit shader is a
        // transparent one that doesn't write depth, so they draw first, as a backdrop everything else draws over
        var backdrop = new Material(unlit) { renderQueue = 1000 };
        Part("walls", surroundings.transform, Walls(), backdrop);
        var ceiling = Part("ceiling", surroundings.transform, Disc(Radius, Ceiling, Ceiling), backdrop);
        ceiling.localPosition = new Vector3(0, Height, 0);

        desk = new GameObject("desk").transform;
        desk.SetParent(deskRoot, false);
        var top = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        Object.Destroy(top.GetComponent<Collider>());
        top.SetParent(desk, false);
        top.localScale = new Vector3(2 * half.x, thickness, 2 * half.y);
        top.localPosition = new Vector3(0, -thickness / 2, centreZ);
        top.GetComponent<Renderer>().sharedMaterial = SimView.Paint(material, SimView.Hex(0x3d3c42), 0.3f);
        var edge = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        Object.Destroy(edge.GetComponent<Collider>());
        edge.SetParent(desk, false);
        edge.localScale = new Vector3(2 * half.x + 0.004f, 0.004f, 0.004f);
        edge.localPosition = new Vector3(0, -0.002f, centreZ - half.y);
        edge.GetComponent<Renderer>().sharedMaterial = SimView.Paint(material, Ui.Accent, 0.6f);
        var legMaterial = SimView.Paint(material, SimView.Hex(0x5a5961), 0.6f);
        for (int i = 0; i < 4; i++)
        {
            legs[i] = GameObject.CreatePrimitive(PrimitiveType.Cylinder).transform;
            Object.Destroy(legs[i].GetComponent<Collider>());
            legs[i].SetParent(desk, false);
            legs[i].GetComponent<Renderer>().sharedMaterial = legMaterial;
        }

        // The light hangs over the desk and moves with it
        lamp = Part("lamp", desk, Lamp(), new Material(unlit) { renderQueue = 1001 });
        lamp.localPosition = new Vector3(0, 0, centreZ);
    }

    // Passthrough shows the real room instead
    public void SetVisible(bool on)
    {
        surroundings.SetActive(on);
        lamp.gameObject.SetActive(on);
    }

    public void SetHeight(float height)
    {
        for (int i = 0; i < 4; i++)
        {
            float x = (i % 2 == 0 ? -1 : 1) * (half.x - 0.05f), z = centreZ + (i < 2 ? -1 : 1) * (half.y - 0.05f);
            legs[i].localScale = new Vector3(0.035f, height / 2, 0.035f);
            legs[i].localPosition = new Vector3(x, -height / 2, z);
        }
        lamp.localPosition = new Vector3(0, Height - height - 0.01f, centreZ);
    }

    static Transform Part(string name, Transform parent, Mesh mesh, Material material)
    {
        var go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetParent(parent, false);
        go.GetComponent<MeshFilter>().sharedMesh = mesh;
        go.GetComponent<MeshRenderer>().sharedMaterial = material;
        return go.transform;
    }

    // Lighter around eye height, darker at the skirting and towards the ceiling
    static Mesh Walls()
    {
        const int n = 72;
        float[] heights = { 0, 0.1f, 0.1f, 0.8f, 1.3f, Height };
        Color[] colors = { Skirting, Skirting, WallLow, WallLow, WallEye, WallTop };
        var v = new Vector3[(n + 1) * heights.Length];
        var c = new Color[v.Length];
        var t = new int[n * (heights.Length - 1) * 6];
        int k = 0;
        for (int i = 0; i <= n; i++)
        {
            float a = 2 * Mathf.PI * i / n;
            for (int j = 0; j < heights.Length; j++)
            {
                v[i * heights.Length + j] = new Vector3(Radius * Mathf.Cos(a), heights[j], Radius * Mathf.Sin(a));
                c[i * heights.Length + j] = colors[j];
            }
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < heights.Length - 1; j++)
            {
                int a = i * heights.Length + j, b = a + heights.Length;
                t[k++] = a; t[k++] = a + 1; t[k++] = b;
                t[k++] = b; t[k++] = a + 1; t[k++] = b + 1;
            }
        var mesh = new Mesh { vertices = v, colors = c, triangles = t };
        mesh.RecalculateBounds();
        return mesh;
    }

    // A flat disc facing down, from the centre colour to the rim colour
    static Mesh Disc(float r, Color centre, Color rim, float inner = 0)
    {
        const int n = 72;
        var v = new Vector3[2 * (n + 1)];
        var c = new Color[v.Length];
        var t = new int[n * 6];
        for (int i = 0; i <= n; i++)
        {
            float a = 2 * Mathf.PI * i / n;
            v[2 * i] = new Vector3(inner * Mathf.Cos(a), 0, inner * Mathf.Sin(a));
            v[2 * i + 1] = new Vector3(r * Mathf.Cos(a), 0, r * Mathf.Sin(a));
            c[2 * i] = centre;
            c[2 * i + 1] = rim;
        }
        for (int i = 0, k = 0; i < n; i++)
        {
            int a = 2 * i, b = 2 * i + 2;
            t[k++] = a; t[k++] = a + 1; t[k++] = b;
            t[k++] = b; t[k++] = a + 1; t[k++] = b + 1;
        }
        var mesh = new Mesh { vertices = v, colors = c, triangles = t };
        mesh.RecalculateBounds();
        return mesh;
    }

    // A round panel with a soft glow on the ceiling around it
    static Mesh Lamp()
    {
        var face = Disc(0.35f, LampFace, LampFace);
        var glow = Disc(1.6f, LampGlow, Ceiling, 0.35f);
        var mesh = new Mesh();
        // Glow first, the panel over it: without depth the order within the mesh decides
        mesh.CombineMeshes(new[]
        {
            new CombineInstance { mesh = glow, transform = Matrix4x4.identity },
            new CombineInstance { mesh = face, transform = Matrix4x4.identity },
        });
        return mesh;
    }

    // Lighter where the user sits, fading out towards the walls, with a faint half-metre grid that fades with it
    static Texture2D FloorTexture()
    {
        const int n = 1024;
        var tex = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Clamp, anisoLevel = 8 };
        var pixels = new Color32[n * n];
        float metre = n / (2 * Radius);
        int cell = Mathf.RoundToInt(metre / 2);
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float dx = (x - n / 2f) / metre, dy = (y - n / 2f) / metre;
                float fade = Mathf.SmoothStep(0, 1, Mathf.Sqrt(dx * dx + dy * dy) / (Radius - 0.5f));
                var col = Color.Lerp(FloorCentre, FloorEdge, fade);
                if (x % cell < 2 || y % cell < 2) col = Color.Lerp(col, WallEye * 1.25f, 0.5f * (1 - fade));
                pixels[y * n + x] = col;
            }
        tex.SetPixels32(pixels);
        tex.Apply(true);
        return tex;
    }
}
