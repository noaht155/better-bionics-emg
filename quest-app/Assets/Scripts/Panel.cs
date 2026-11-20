using System;
using System.Collections.Generic;
using UnityEngine;

// Settings panel in the scene, worked with the right controller's ray and trigger. Rows of a label and buttons; the
// app sets the labels each frame.
public class Panel
{
    const float RowH = 0.05f;
    const float Width = 0.34f;
    static readonly Color Back = SimView.Hex(0x0a0a0b), ButtonColor = SimView.Hex(0x27272a),
        HoverColor = SimView.Hex(0x3f3f46), TextColor = SimView.Hex(0xf5f5f5), Accent = SimView.Hex(0xff3d8b);

    class Button
    {
        public Collider collider;
        public Material material;
        public Action click;
    }

    public readonly Transform root;
    readonly Material material;
    readonly Font font;
    readonly List<Button> buttons = new List<Button>();
    Button hovered;
    int rows;

    public Panel(Transform parent, Material material)
    {
        this.material = material;
        font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        root = new GameObject("panel").transform;
        root.SetParent(parent, false);
    }

    public TextMesh Row(string label, params (string text, Action click)[] actions)
    {
        float y = -rows++ * RowH;
        var text = Text(new Vector3(-Width / 2 + 0.015f, y, -0.006f), label, TextAnchor.MiddleLeft);
        float x = Width / 2 - 0.03f;
        for (int i = actions.Length - 1; i >= 0; i--)
        {
            float w = Mathf.Max(0.04f, 0.012f * actions[i].text.Length + 0.016f);
            x -= w / 2;
            AddButton(new Vector3(x, y, -0.004f), new Vector2(w, RowH * 0.75f), actions[i].text, actions[i].click);
            x -= w / 2 + 0.008f;
        }
        return text;
    }

    // Puts the background behind the rows once they're all added
    public void Finish()
    {
        float h = rows * RowH + 0.02f;
        var back = Box(new Vector3(0, -(rows - 1) * RowH / 2, 0), new Vector3(Width, h, 0.004f), Back);
        UnityEngine.Object.Destroy(back.GetComponent<Collider>());
    }

    TextMesh Text(Vector3 pos, string s, TextAnchor anchor)
    {
        var go = new GameObject("text", typeof(MeshRenderer));
        go.transform.SetParent(root, false);
        go.transform.localPosition = pos;
        var t = go.AddComponent<TextMesh>();
        t.font = font;
        t.fontSize = 64;
        t.characterSize = 0.0025f;
        t.anchor = anchor;
        t.color = TextColor;
        t.text = s;
        go.GetComponent<MeshRenderer>().sharedMaterial = font.material;
        return t;
    }

    GameObject Box(Vector3 pos, Vector3 size, Color c)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.transform.SetParent(root, false);
        go.transform.localPosition = pos;
        go.transform.localScale = size;
        go.GetComponent<Renderer>().sharedMaterial = new Material(material) { color = c };
        return go;
    }

    void AddButton(Vector3 pos, Vector2 size, string label, Action click)
    {
        var box = Box(pos, new Vector3(size.x, size.y, 0.006f), ButtonColor);
        Text(pos + new Vector3(0, 0, -0.004f), label, TextAnchor.MiddleCenter);
        buttons.Add(new Button { collider = box.GetComponent<Collider>(), material = box.GetComponent<Renderer>().sharedMaterial, click = click });
    }

    // Highlights what the ray points at and clicks it on a trigger press; returns the ray's length, or 0 when it
    // misses the panel
    public float Point(Ray ray, bool pressed)
    {
        Button hit = null;
        float length = 0;
        if (Physics.Raycast(ray, out var info, 3f))
        {
            hit = buttons.Find(b => b.collider == info.collider);
            length = info.distance;
        }
        if (hovered != hit)
        {
            if (hovered != null) hovered.material.color = ButtonColor;
            if (hit != null) hit.material.color = HoverColor;
            hovered = hit;
        }
        if (pressed && hit != null)
        {
            hit.material.color = Accent;
            hit.click();
        }
        return hit != null ? length : 0;
    }
}
