using UnityEngine;

// The room around the sim: a round room with warm walls, a wooden floor, a ceiling light over the desk and a few
// things around the desk (rug, plant, sideboard with books, floor lamp), and the desk itself, sized like the sim's desk
// (export_model.py) with its top at the desk root and legs down to the real floor. Dim and warm (Noah, 2026-11-24:
// the first version felt clinical, empty and a bit bright).
public class Room
{
    const float Radius = 4, Height = 3;
    readonly Transform desk, lamp, furnishings;
    readonly GameObject surroundings;
    readonly Transform[] legs = new Transform[4];
    readonly Vector2 half;
    readonly float centreZ;

    static readonly Color Skirting = SimView.Hex(0x221b16), WallLow = SimView.Hex(0x30271f),
        WallEye = SimView.Hex(0x3b3027), WallTop = SimView.Hex(0x1f1a15), Ceiling = SimView.Hex(0x1c1814),
        LampFace = SimView.Hex(0xf3dcb6), LampGlow = SimView.Hex(0x3e342b), Wood = SimView.Hex(0x3a2a1f),
        WoodDark = SimView.Hex(0x2b1f17);

    public Room(Transform deskRoot, Material material, Material unlit, Vector2 deskHalf, float deskCentreY,
        float thickness)
    {
        half = deskHalf;
        centreZ = -deskCentreY;
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Exponential;
        RenderSettings.fogDensity = 0.06f;
        RenderSettings.fogColor = WallLow;

        surroundings = new GameObject("surroundings");
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "floor";
        Object.Destroy(floor.GetComponent<Collider>());
        floor.transform.SetParent(surroundings.transform, false);
        // The plane is 10 m, 2 Radius across here so it ends under the wall
        floor.transform.localScale = new Vector3(2 * Radius / 10, 1, 2 * Radius / 10);
        var fm = SimView.Paint(material, Color.white, 0.2f);
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
        var top = Box(desk, new Vector3(2 * half.x, thickness, 2 * half.y), new Vector3(0, -thickness / 2, centreZ),
            SimView.Paint(material, SimView.Hex(0x4b3a2e), 0.35f));
        top.name = "desk top";
        Box(desk, new Vector3(2 * half.x + 0.004f, 0.004f, 0.004f), new Vector3(0, -0.002f, centreZ - half.y),
            SimView.Paint(material, Ui.Accent, 0.6f));
        var legMaterial = SimView.Paint(material, SimView.Hex(0x2e2a28), 0.5f);
        for (int i = 0; i < 4; i++)
        {
            legs[i] = GameObject.CreatePrimitive(PrimitiveType.Cylinder).transform;
            Object.Destroy(legs[i].GetComponent<Collider>());
            legs[i].SetParent(desk, false);
            legs[i].GetComponent<Renderer>().sharedMaterial = legMaterial;
        }

        // The light hangs over the desk and the furnishings stand around it, so they move with it
        lamp = Part("lamp", desk, Lamp(), new Material(unlit) { renderQueue = 1001 });
        lamp.localPosition = new Vector3(0, 0, centreZ);
        furnishings = new GameObject("furnishings").transform;
        furnishings.SetParent(desk, false);
        Furnish(material);
    }

    // Passthrough shows the real room instead
    public void SetVisible(bool on)
    {
        surroundings.SetActive(on);
        lamp.gameObject.SetActive(on);
        furnishings.gameObject.SetActive(on);
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
        furnishings.localPosition = new Vector3(0, -height, centreZ);
    }

    // Around the desk, in the desk's frame with the floor at y 0 and the desk's middle at z 0: the user sits at -z,
    // +x is their right
    void Furnish(Material material)
    {
        var t = furnishings;
        // Rug under the desk and chair, a border round a darker middle
        Box(t, new Vector3(2.3f, 0.004f, 1.9f), new Vector3(0, 0.002f, -0.25f), SimView.Paint(material, SimView.Hex(0x5a3d31), 0.05f));
        Box(t, new Vector3(2.1f, 0.006f, 1.7f), new Vector3(0, 0.003f, -0.25f), SimView.Paint(material, SimView.Hex(0x3d2a24), 0.05f));

        // Plant on the right, past the desk
        var pot = SimView.Paint(material, SimView.Hex(0x8a5238), 0.3f);
        Shape(t, PrimitiveType.Cylinder, new Vector3(0.28f, 0.16f, 0.28f), new Vector3(1.15f, 0.16f, 0.55f), pot);
        Shape(t, PrimitiveType.Cylinder, new Vector3(0.25f, 0.01f, 0.25f), new Vector3(1.15f, 0.315f, 0.55f),
            SimView.Paint(material, SimView.Hex(0x2a1d15), 0));
        var leafA = SimView.Paint(material, SimView.Hex(0x3e6a45), 0.35f);
        var leafB = SimView.Paint(material, SimView.Hex(0x507e52), 0.35f);
        var stem = SimView.Paint(material, SimView.Hex(0x3a4a2c), 0.2f);
        for (int i = 0; i < 9; i++)
        {
            float a = i * 2.4f, h = 0.5f + 0.06f * (i % 4), r = 0.08f + 0.03f * (i % 3);
            var at = new Vector3(1.15f + r * Mathf.Cos(a), h, 0.55f + r * Mathf.Sin(a));
            var s = Shape(t, PrimitiveType.Cylinder, new Vector3(0.012f, (h - 0.3f) / 2, 0.012f),
                new Vector3((1.15f + at.x) / 2, (0.3f + h) / 2, (0.55f + at.z) / 2), stem);
            s.localRotation = Quaternion.FromToRotation(Vector3.up, at - new Vector3(1.15f, 0.3f, 0.55f));
            var leaf = Shape(t, PrimitiveType.Sphere, new Vector3(0.2f, 0.03f, 0.09f), at, i % 2 == 0 ? leafA : leafB);
            leaf.localRotation = Quaternion.Euler(-25 + 10 * (i % 3), a * Mathf.Rad2Deg, 15);
        }

        // Low sideboard on the left with books and a vase on it
        var wood = SimView.Paint(material, SimView.Hex(0x5a3e2b), 0.3f);
        var dark = SimView.Paint(material, SimView.Hex(0x3a281c), 0.3f);
        Box(t, new Vector3(1.3f, 0.7f, 0.42f), new Vector3(-1.45f, 0.38f, 0.8f), wood);
        Box(t, new Vector3(1.34f, 0.03f, 0.46f), new Vector3(-1.45f, 0.745f, 0.8f), dark);
        Box(t, new Vector3(1.22f, 0.006f, 0.005f), new Vector3(-1.45f, 0.4f, 0.588f), dark);
        Color[] spines = { SimView.Hex(0x7a5c4a), SimView.Hex(0x4c5a66), SimView.Hex(0x8a7a5a), SimView.Hex(0x6a4a4a), SimView.Hex(0x51604c) };
        float x = -1.95f;
        for (int i = 0; i < spines.Length; i++)
        {
            float w = 0.04f + 0.012f * (i % 3), h = 0.2f + 0.03f * ((i * 2) % 3);
            Box(t, new Vector3(w, h, 0.16f), new Vector3(x + w / 2, 0.76f + h / 2, 0.8f), SimView.Paint(material, spines[i], 0.2f));
            x += w + 0.004f;
        }
        Shape(t, PrimitiveType.Cylinder, new Vector3(0.11f, 0.11f, 0.11f), new Vector3(-1.0f, 0.87f, 0.82f),
            SimView.Paint(material, SimView.Hex(0x9a8f80), 0.6f));

        // Floor lamp behind on the left, its shade lit from inside
        Shape(t, PrimitiveType.Cylinder, new Vector3(0.26f, 0.012f, 0.26f), new Vector3(-1.2f, 0.012f, -1.0f), dark);
        Shape(t, PrimitiveType.Cylinder, new Vector3(0.02f, 0.72f, 0.02f), new Vector3(-1.2f, 0.74f, -1.0f), dark);
        var shade = SimView.Paint(material, SimView.Hex(0xe8c48e), 0.1f);
        shade.EnableKeyword("_EMISSION");
        shade.SetColor("_EmissionColor", SimView.Hex(0xb08550));
        Shape(t, PrimitiveType.Cylinder, new Vector3(0.34f, 0.14f, 0.34f), new Vector3(-1.2f, 1.5f, -1.0f), shade);
    }

    static Transform Shape(Transform parent, PrimitiveType type, Vector3 scale, Vector3 pos, Material paint)
    {
        var go = GameObject.CreatePrimitive(type);
        Object.Destroy(go.GetComponent<Collider>());
        go.transform.SetParent(parent, false);
        go.transform.localScale = scale;
        go.transform.localPosition = pos;
        go.GetComponent<Renderer>().sharedMaterial = paint;
        return go.transform;
    }

    static Transform Box(Transform parent, Vector3 size, Vector3 pos, Material paint) =>
        Shape(parent, PrimitiveType.Cube, size, pos, paint);

    static Transform Part(string name, Transform parent, Mesh mesh, Material material)
    {
        var go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetParent(parent, false);
        go.GetComponent<MeshFilter>().sharedMesh = mesh;
        go.GetComponent<MeshRenderer>().sharedMaterial = material;
        return go.transform;
    }

    // A darker band up to the dado rail, lighter above around eye height, darker towards the ceiling
    static Mesh Walls()
    {
        const int n = 72;
        float[] heights = { 0, 0.1f, 0.1f, 0.9f, 0.9f, 0.93f, 0.93f, 1.4f, Height };
        Color[] colors = { Skirting, Skirting, WallLow, WallLow, WallTop, WallTop, WallEye, WallEye, WallTop };
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
        var face = Disc(0.3f, LampFace, LampFace);
        var glow = Disc(1.2f, LampGlow, Ceiling, 0.3f);
        var mesh = new Mesh();
        // Glow first, the panel over it: without depth the order within the mesh decides
        mesh.CombineMeshes(new[]
        {
            new CombineInstance { mesh = glow, transform = Matrix4x4.identity },
            new CombineInstance { mesh = face, transform = Matrix4x4.identity },
        });
        return mesh;
    }

    // Wooden boards, 18 cm wide and 1.2 to 2.4 m long with staggered ends, darker towards the walls
    static Texture2D FloorTexture()
    {
        const int n = 1024;
        var tex = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Clamp, anisoLevel = 8 };
        var pixels = new Color32[n * n];
        float metre = n / (2 * Radius);
        int board = Mathf.RoundToInt(0.18f * metre);
        var random = new System.Random(7);
        int rows = n / board + 1;
        var offsets = new float[rows];
        var lengths = new float[rows];
        var shades = new float[rows * 8];
        for (int r = 0; r < rows; r++)
        {
            offsets[r] = (float)random.NextDouble() * 2.4f;
            lengths[r] = 1.2f + 1.2f * (float)random.NextDouble();
        }
        for (int i = 0; i < shades.Length; i++) shades[i] = 0.85f + 0.3f * (float)random.NextDouble();
        for (int y = 0; y < n; y++)
        {
            int row = y / board;
            bool seam = y % board < 2;
            for (int x = 0; x < n; x++)
            {
                float along = x / metre + offsets[row];
                int plank = Mathf.FloorToInt(along / lengths[row]);
                bool end = along - plank * lengths[row] < 2 / metre;
                float shade = shades[(row * 3 + plank) % shades.Length];
                // Grain: a slow ripple along the board
                float grain = 0.94f + 0.06f * Mathf.Sin(along * 23 + row * 1.7f + Mathf.Sin(along * 3.1f) * 2);
                var col = seam || end ? WoodDark * 0.8f : Wood * (shade * grain);
                float dx = (x - n / 2f) / metre, dy = (y - n / 2f) / metre;
                float fade = Mathf.SmoothStep(0, 1, Mathf.Sqrt(dx * dx + dy * dy) / (Radius - 0.3f));
                col = Color.Lerp(col, WoodDark * 0.7f, 0.6f * fade);
                col.a = 1;
                pixels[y * n + x] = col;
            }
        }
        tex.SetPixels32(pixels);
        tex.Apply(true);
        return tex;
    }
}
