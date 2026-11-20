using UnityEngine;

// The room around the sim: a floor with a faint grid fading into the dark, and the desk, sized like the sim's desk
// (export_model.py) with its top at the desk root and legs down to the real floor.
public class Room
{
    readonly Transform desk;
    readonly Transform[] legs = new Transform[4];
    readonly Vector2 half;
    readonly float centreZ;

    public Room(Transform deskRoot, Material material, Vector2 deskHalf, float deskCentreY, float thickness)
    {
        half = deskHalf;
        centreZ = -deskCentreY;
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Exponential;
        RenderSettings.fogDensity = 0.12f;
        RenderSettings.fogColor = Ui.Bg;

        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "floor";
        Object.Destroy(floor.GetComponent<Collider>());
        floor.transform.localScale = new Vector3(4, 1, 4);
        var fm = SimView.Paint(material, Color.white, 0.1f);
        fm.mainTexture = Grid();
        // Half-metre squares over the 40 m floor
        fm.mainTextureScale = new Vector2(80, 80);
        floor.GetComponent<Renderer>().sharedMaterial = fm;

        desk = new GameObject("desk").transform;
        desk.SetParent(deskRoot, false);
        var top = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        Object.Destroy(top.GetComponent<Collider>());
        top.SetParent(desk, false);
        top.localScale = new Vector3(2 * half.x, thickness, 2 * half.y);
        top.localPosition = new Vector3(0, -thickness / 2, centreZ);
        top.GetComponent<Renderer>().sharedMaterial = SimView.Paint(material, SimView.Hex(0x2a2a2e), 0.35f);
        var edge = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        Object.Destroy(edge.GetComponent<Collider>());
        edge.SetParent(desk, false);
        edge.localScale = new Vector3(2 * half.x + 0.004f, 0.004f, 0.004f);
        edge.localPosition = new Vector3(0, -0.002f, centreZ - half.y);
        edge.GetComponent<Renderer>().sharedMaterial = SimView.Paint(material, Ui.Accent, 0.6f);
        var legMaterial = SimView.Paint(material, SimView.Hex(0x3f3f46), 0.7f);
        for (int i = 0; i < 4; i++)
        {
            legs[i] = GameObject.CreatePrimitive(PrimitiveType.Cylinder).transform;
            Object.Destroy(legs[i].GetComponent<Collider>());
            legs[i].SetParent(desk, false);
            legs[i].GetComponent<Renderer>().sharedMaterial = legMaterial;
        }
    }

    public void SetHeight(float height)
    {
        for (int i = 0; i < 4; i++)
        {
            float x = (i % 2 == 0 ? -1 : 1) * (half.x - 0.05f), z = centreZ + (i < 2 ? -1 : 1) * (half.y - 0.05f);
            legs[i].localScale = new Vector3(0.035f, height / 2, 0.035f);
            legs[i].localPosition = new Vector3(x, -height / 2, z);
        }
    }

    static Texture2D Grid()
    {
        const int n = 128;
        var tex = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Repeat, anisoLevel = 8 };
        var bg = SimView.Hex(0x111113);
        var line = SimView.Hex(0x232327);
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
                tex.SetPixel(x, y, x < 2 || y < 2 ? line : bg);
        tex.Apply(true);
        return tex;
    }
}
