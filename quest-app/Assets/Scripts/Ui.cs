using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

// The menu: a floating panel with tabs, worked with the free hand's ray and trigger. Colours from the app's theme
// (visual-style tokens), text in TextMeshPro so it stays sharp in the headset. Sizes in metres.
public class Ui
{
    public const float Width = 0.42f;
    const float Pad = 0.018f, RowH = 0.042f, TabH = 0.04f, Corner = 0.008f;
    public static readonly Color Bg = SimView.Hex(0x0a0a0b), Raised = SimView.Hex(0x151517), Line = SimView.Hex(0x27272a),
        LineStrong = SimView.Hex(0x3f3f46), Text = SimView.Hex(0xf5f5f5), Muted = SimView.Hex(0xa1a1aa),
        Accent = SimView.Hex(0xff3d8b), Good = SimView.Hex(0x3ddc84);

    class Button
    {
        public Collider collider;
        public Material material;
        public Color color;
        public Action click;
        public Func<bool> on;
    }

    public class Page
    {
        public Transform root;
        public float y;
    }

    public readonly Transform root;
    readonly Material material;
    readonly Transform body;
    readonly List<Button> buttons = new List<Button>();
    readonly List<Page> pages = new List<Page>();
    readonly List<(Button b, Page p)> tabs = new List<(Button, Page)>();
    Button hovered;
    Page shown;
    Transform back;
    float height;

    public bool Visible => root.gameObject.activeSelf;

    public Ui(Material material)
    {
        this.material = material;
        root = new GameObject("menu").transform;
        body = new GameObject("body").transform;
        body.SetParent(root, false);
    }

    public Page AddPage(string title)
    {
        var page = new Page { root = new GameObject(title).transform, y = -(Pad + TabH + Pad) };
        page.root.SetParent(body, false);
        pages.Add(page);
        float w = (Width - 2 * Pad - (pages.Count - 1) * 0.006f) / pages.Count;
        var tab = AddButton(body, Vector3.zero, new Vector2(w, TabH), title, () => Show(page), () => shown == page);
        tabs.Add((tab, page));
        for (int i = 0; i < tabs.Count; i++)
        {
            float tw = (Width - 2 * Pad - (tabs.Count - 1) * 0.006f) / tabs.Count;
            tabs[i].b.collider.transform.parent.localPosition =
                new Vector3(-Width / 2 + Pad + tw / 2 + i * (tw + 0.006f), -Pad - TabH / 2, 0);
            Resize(tabs[i].b, new Vector2(tw, TabH));
        }
        return page;
    }

    public void Show(Page page)
    {
        foreach (var p in pages) p.root.gameObject.SetActive(p == page);
        shown = page;
        Layout();
    }

    public TextMeshPro Heading(Page page, string text)
    {
        page.y -= 0.006f;
        var t = Label(page.root, new Vector3(-Width / 2 + Pad, page.y - RowH * 0.35f, 0), text, 0.013f, Muted,
            TextAlignmentOptions.Left);
        page.y -= RowH * 0.7f;
        return t;
    }

    // A label on the left and buttons on the right; returns the label to update
    public TextMeshPro Row(Page page, string label, params (string text, Action click)[] actions) =>
        Row(page, label, null, actions);

    public TextMeshPro Row(Page page, string label, Func<bool> on, params (string text, Action click)[] actions)
    {
        float y = page.y - RowH / 2;
        var t = Label(page.root, new Vector3(-Width / 2 + Pad, y, 0), label, 0.016f, Text, TextAlignmentOptions.Left);
        float x = Width / 2 - Pad;
        for (int i = actions.Length - 1; i >= 0; i--)
        {
            float w = Mathf.Max(0.04f, 0.0105f * actions[i].text.Length + 0.024f);
            x -= w / 2;
            AddButton(page.root, new Vector3(x, y, 0), new Vector2(w, RowH * 0.8f), actions[i].text, actions[i].click,
                actions.Length == 1 ? on : null);
            x -= w / 2 + 0.006f;
        }
        page.y -= RowH;
        return t;
    }

    // A row of equal buttons across the panel
    public void Buttons(Page page, params (string text, Action click)[] actions)
    {
        float y = page.y - RowH / 2;
        float w = (Width - 2 * Pad - (actions.Length - 1) * 0.006f) / actions.Length;
        for (int i = 0; i < actions.Length; i++)
            AddButton(page.root, new Vector3(-Width / 2 + Pad + w / 2 + i * (w + 0.006f), y, 0), new Vector2(w, RowH * 0.8f),
                actions[i].text, actions[i].click);
        page.y -= RowH;
    }

    public TextMeshPro Note(Page page, int lines = 1)
    {
        var t = Label(page.root, new Vector3(-Width / 2 + Pad, page.y - 0.008f, 0), "", 0.013f, Muted,
            TextAlignmentOptions.TopLeft);
        t.rectTransform.pivot = new Vector2(0, 1);
        t.rectTransform.sizeDelta = new Vector2(Width - 2 * Pad, lines * 0.02f);
        t.textWrappingMode = TextWrappingModes.Normal;
        page.y -= lines * 0.02f + 0.012f;
        return t;
    }

    void Layout()
    {
        height = -(shown?.y ?? 0) + Pad;
        if (back != null) UnityEngine.Object.Destroy(back.gameObject);
        back = Quad(body, new Vector3(0, -height / 2, 0.004f), new Vector2(Width, height), Bg, Corner * 1.5f, LineStrong).transform;
        body.localPosition = new Vector3(0, height / 2, 0);
    }

    // Opens in front of the head, a little below the eyes, turned to face it
    public void Open(Transform head)
    {
        var forward = Vector3.ProjectOnPlane(head.forward, Vector3.up).normalized;
        if (forward.sqrMagnitude < 0.5f) forward = Vector3.forward;
        root.position = head.position + forward * 0.55f + Vector3.down * 0.12f;
        root.rotation = Quaternion.LookRotation(root.position - head.position, Vector3.up);
        root.gameObject.SetActive(true);
    }

    public void Close() => root.gameObject.SetActive(false);

    TextMeshPro Label(Transform parent, Vector3 pos, string text, float size, Color color, TextAlignmentOptions align)
    {
        var go = new GameObject("text");
        go.transform.SetParent(parent, false);
        var t = go.AddComponent<TextMeshPro>();
        t.text = text;
        // TextMeshPro's world text is about a tenth of a unit tall per point
        t.fontSize = size * 10;
        t.color = color;
        t.alignment = align;
        t.textWrappingMode = TextWrappingModes.NoWrap;
        t.rectTransform.sizeDelta = new Vector2(Width * 0.6f, RowH);
        t.rectTransform.pivot = new Vector2(align == TextAlignmentOptions.Center ? 0.5f : 0, 0.5f);
        go.transform.localPosition = pos + new Vector3(0, 0, -0.002f);
        return t;
    }

    Button AddButton(Transform parent, Vector3 pos, Vector2 size, string text, Action click, Func<bool> on = null)
    {
        var holder = new GameObject("button").transform;
        holder.SetParent(parent, false);
        holder.localPosition = pos;
        var quad = Quad(holder, Vector3.zero, size, Raised, Corner, Line);
        var collider = quad.AddComponent<BoxCollider>();
        collider.size = new Vector3(size.x, size.y, 0.01f);
        var label = Label(holder, Vector3.zero, text, 0.015f, Text, TextAlignmentOptions.Center);
        label.rectTransform.sizeDelta = size;
        var b = new Button
        {
            collider = collider, material = quad.GetComponent<Renderer>().sharedMaterial, color = Raised, click = click, on = on
        };
        buttons.Add(b);
        return b;
    }

    void Resize(Button b, Vector2 size)
    {
        var quad = b.collider.gameObject;
        quad.GetComponent<MeshFilter>().sharedMesh = RoundedRect(size, Corner);
        ((BoxCollider)b.collider).size = new Vector3(size.x, size.y, 0.01f);
        quad.transform.parent.GetComponentInChildren<TextMeshPro>().rectTransform.sizeDelta = size;
    }

    GameObject Quad(Transform parent, Vector3 pos, Vector2 size, Color color, float corner, Color border)
    {
        var go = new GameObject("quad", typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetParent(parent, false);
        go.transform.localPosition = pos;
        go.GetComponent<MeshFilter>().sharedMesh = RoundedRect(size, corner);
        go.GetComponent<Renderer>().sharedMaterial = new Material(material) { color = color };
        var edge = new GameObject("edge", typeof(MeshFilter), typeof(MeshRenderer));
        edge.transform.SetParent(go.transform, false);
        edge.transform.localPosition = new Vector3(0, 0, 0.0005f);
        edge.GetComponent<MeshFilter>().sharedMesh = RoundedRect(size + Vector2.one * 0.002f, corner + 0.001f);
        edge.GetComponent<Renderer>().sharedMaterial = new Material(material) { color = border };
        return go;
    }

    // A flat rounded rectangle facing -z (towards the user), as a triangle fan round its centre
    static Mesh RoundedRect(Vector2 size, float r)
    {
        const int arc = 6;
        r = Mathf.Min(r, size.x / 2, size.y / 2);
        var v = new List<Vector3> { Vector3.zero };
        Vector2 half = size / 2 - Vector2.one * r;
        Vector2[] corners = { new Vector2(half.x, half.y), new Vector2(-half.x, half.y), new Vector2(-half.x, -half.y), new Vector2(half.x, -half.y) };
        for (int c = 0; c < 4; c++)
            for (int i = 0; i <= arc; i++)
            {
                float a = Mathf.PI / 2 * (c + (float)i / arc);
                v.Add(new Vector3(corners[c].x + r * Mathf.Cos(a), corners[c].y + r * Mathf.Sin(a), 0));
            }
        var t = new List<int>();
        for (int i = 1; i < v.Count; i++)
        {
            int next = i + 1 < v.Count ? i + 1 : 1;
            t.AddRange(new[] { 0, next, i });
        }
        var mesh = new Mesh();
        mesh.SetVertices(v);
        mesh.SetTriangles(t, 0);
        mesh.RecalculateNormals();
        return mesh;
    }

    // Highlights what the ray points at and clicks it on a trigger press; returns the ray's length, 0 when it misses
    public float Point(Ray ray, bool pressed, out bool clicked)
    {
        clicked = false;
        Button hit = null;
        float length = 0;
        if (Visible && Physics.Raycast(ray, out var info, 3f))
        {
            hit = buttons.Find(b => b.collider == info.collider);
            length = info.distance;
        }
        hovered = hit;
        if (pressed && hit != null)
        {
            hit.click();
            clicked = true;
        }
        foreach (var b in buttons)
        {
            bool on = b.on != null && b.on();
            var c = on ? Color.Lerp(Raised, Accent, 0.55f) : Raised;
            if (b == hovered) c = Color.Lerp(c, LineStrong, 0.7f);
            if (b.material.color != c) b.material.color = c;
        }
        return hit != null ? length : 0;
    }
}
