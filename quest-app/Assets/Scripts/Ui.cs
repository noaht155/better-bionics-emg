using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

// The headset's menu: the web app's look (its colour tokens, Inter and JetBrains Mono, numbered sections with the
// dashed line and pink dot, gradient highlights) on a glass panel you touch with a fingertip or point at with a
// controller. Every part is a flat mesh with vertex colours on one transparent material, drawn in layers by
// sortingOrder. Sizes in metres; the panel's front faces -z.
public class Ui
{
    public const float Width = 0.40f;
    const float Pad = 0.02f, RowH = 0.04f, Corner = 0.012f;
    public static readonly Color Bg = SimView.Hex(0x0a0a0b), Raised = SimView.Hex(0x151517), Line = SimView.Hex(0x27272a),
        LineStrong = SimView.Hex(0x3f3f46), Text = SimView.Hex(0xf5f5f5), Muted = SimView.Hex(0xa1a1aa),
        Accent = SimView.Hex(0xff3d8b), Orange = SimView.Hex(0xff7a3d), Magenta = SimView.Hex(0xc93ee6),
        Good = SimView.Hex(0x3ddc84), Warn = SimView.Hex(0xffc233), Bad = SimView.Hex(0xff3b3b);
    // Layers, back to front
    const int LBack = 0, LEdge = 1, LControl = 2, LFill = 3, LKnob = 4, LText = 5;
    // Motion only where something changes state, all under 200 ms as on the web app: a toggle's pill slides, the
    // tab underline slides to the tab picked, the panel grows in from 92 % when it opens
    const float SlideS = 0.14f, OpenS = 0.16f;
    // A press that has to be held (through a ray) fires after this long, its bar filling under the button meanwhile
    // (Noah, 2026-11-24: pinches set things off by accident)
    public const float HoldS = 1.0f;
    // One press per pinch or pull: after a press, another only starts once nothing has touched for this long, so a
    // pinch that flickers for a frame while held doesn't select again (Noah, 2026-11-24)
    const float ReleaseS = 0.2f;

    public static Font InterFont, InterBoldFont, MonoFont;
    static TMP_FontAsset inter, interBold, mono;

    enum Kind { Button, Toggle, Slider, Handle }

    class Element
    {
        public Kind kind;
        public Vector2 centre, size;
        public Page page;
        public Action click;
        public Func<bool> on;
        public Func<float> get;
        public Action<float> set;
        public float min, max, step;
        public bool primary;
        public MeshFilter face, edge, fill, knob, bar;
        public TextMeshPro label;
        public float hover, press, slide = -1, hold, pulse;
        public bool fired;
        // What the laptop's VR tab calls it: a toggle's or slider's row label, otherwise the button's text
        public string name;
    }

    // The panel as the laptop's VR tab draws it (vr.js drawPanel): its controls and section titles on the page shown,
    // in metres from the panel's top centre, x right and y up
    [Serializable]
    public class ItemOut
    {
        public string n, k, t;
        public float x, y, w, h, lx, v;
        public bool on;
    }

    [Serializable]
    public class PanelOut
    {
        public string name, page;
        public float w, h;
        public float[] pos = new float[3], right = new float[3], up = new float[3];
        public ItemOut[] items;
        public int hover = -1, press = -1;
        public float hold;
    }

    // While the laptop points at a control, it pulses this long
    const float PulseS = 3;

    public class Page
    {
        public Transform root;
        public float y;
        public int sections;
        // Two columns: the second starts at the first section below the panel's column height, laid out where
        // the first started, and Arrange moves each column to its half once
        public float top, firstEnd;
        public int splitChild = -1, splitElement = -1;
        public bool arranged;
    }

    public readonly Transform root;
    readonly Material material;
    readonly Transform body;
    readonly List<Element> elements = new List<Element>();
    readonly List<(Element tab, Page page)> tabs = new List<(Element, Page)>();
    readonly List<Page> pages = new List<Page>();
    Page shown;
    MeshFilter back, backEdge, topStrip;
    Element pressed, handle, hovered, lastHovered;
    float hoverOff;
    readonly List<(Page page, TextMeshPro label, string title)> sectionLabels = new List<(Page, TextMeshPro, string)>();
    float height;
    float tabsY;
    MeshFilter tabBar;
    float tabFrom, tabTo, tabT = 1, opened = 1;
    bool tabPlaced;
    bool wasTouching, behind, armed = true;
    float releasedFor, offFor;
    float prevZ = -1;
    Vector3 grabOffset;
    Quaternion grabTurn;
    public TextMeshPro chip;
    Element closeButton;
    // The status chip sits left of the close button
    const float ChipInset = 0.035f;
    MeshFilter chipDot;

    public bool Visible => root.gameObject.activeSelf;
    public bool Grabbed => handle != null && pressed == handle;
    // The last Interact moved onto a control it wasn't on before, for a hover tick
    public bool HoverChanged { get; private set; }
    public string PageName => shown != null ? shown.root.name : "";

    readonly bool header;
    // A wide panel has two columns of Width; long pages flow into the second so the panel stays short enough to clear
    // the desk (Noah, 2026-11-24)
    readonly float panelW, columnH;

    // header: the wordmark, status chip and tabs; without it the panel is only its page (the cue card, the dock)
    public Ui(Material material, bool header = true, float columnH = 0)
    {
        this.header = header;
        this.columnH = columnH;
        panelW = columnH > 0 ? 2 * Width : Width;
        this.material = material;
        if (inter == null)
        {
            inter = TMP_FontAsset.CreateFontAsset(InterFont);
            interBold = TMP_FontAsset.CreateFontAsset(InterBoldFont);
            mono = TMP_FontAsset.CreateFontAsset(MonoFont);
        }
        root = new GameObject("menu").transform;
        body = new GameObject("body").transform;
        body.SetParent(root, false);
        tabsY = -Pad - 0.03f;
        if (!header) return;
        var word = Label(body, new Vector3(-panelW / 2 + Pad, -Pad - 0.008f, 0), "BETTER BIONICS", 0.012f, Text, mono,
            TextAlignmentOptions.Left);
        word.characterSpacing = 18;
        word.enableVertexGradient = true;
        word.colorGradient = new VertexGradient(Orange, Magenta, Orange, Magenta);
        // Close in the corner, as on a window (Noah, 2026-11-24: closing by the hand gesture again was a chore)
        var close = new Element { kind = Kind.Button, centre = new Vector2(panelW / 2 - Pad - 0.008f, -Pad - 0.008f),
            size = new Vector2(0.03f, 0.03f), click = Close, name = "Close" };
        close.label = Label(body, new Vector3(close.centre.x, close.centre.y, 0), "×", 0.022f, Muted, inter,
            TextAlignmentOptions.Center);
        close.label.rectTransform.sizeDelta = close.size;
        elements.Add(close);
        closeButton = close;
        chip = Label(body, new Vector3(panelW / 2 - Pad - ChipInset, -Pad - 0.008f, 0), "", 0.011f, Muted, mono, TextAlignmentOptions.Right);
        chip.characterSpacing = 8;
        chip.rectTransform.pivot = new Vector2(1, 0.5f);
        chipDot = Shape(body, LText, Vector3.zero, Vector2.one * 0.008f, 0.004f, Muted);
    }

    public static void SetChip(Ui ui, string text, Color dot)
    {
        ui.chip.text = text;
        float w = ui.chip.preferredWidth;
        ui.chipDot.transform.localPosition = new Vector3(ui.panelW / 2 - Pad - ChipInset - w - 0.01f, -Pad - 0.008f, -0.001f);
        Recolor(ui.chipDot.sharedMesh, dot);
    }

    // Pages and tabs

    public Page AddPage(string title)
    {
        var page = new Page { root = new GameObject(title).transform, y = header ? tabsY - 0.05f : -Pad };
        page.top = page.y;
        page.root.SetParent(body, false);
        pages.Add(page);
        if (!header) return page;
        var tab = new Element { kind = Kind.Button, size = new Vector2(0.06f, 0.03f), click = () => Show(page), name = title };
        tab.label = Label(body, Vector3.zero, title.ToUpperInvariant(), 0.011f, Muted, mono, TextAlignmentOptions.Center);
        tab.label.characterSpacing = 10;
        elements.Add(tab);
        if (tabBar == null) tabBar = Shape(body, LFill, Vector3.zero, new Vector2(0.05f, 0.0025f), 0.001f, Accent, true);
        tabs.Add((tab, page));
        float w = (panelW - 2 * Pad) / tabs.Count;
        for (int i = 0; i < tabs.Count; i++)
        {
            var t = tabs[i].tab;
            t.centre = new Vector2(-panelW / 2 + Pad + w * (i + 0.5f), tabsY - 0.012f);
            t.size = new Vector2(w, 0.03f);
            t.label.transform.localPosition = new Vector3(t.centre.x, t.centre.y, -0.002f);
            t.label.rectTransform.sizeDelta = t.size;
        }

        tabBar.sharedMesh = GradientBar(new Vector2(w * 0.7f, 0.0025f));
        return page;
    }

    public void Show(Page page)
    {
        foreach (var p in pages) Arrange(p);
        foreach (var p in pages) p.root.gameObject.SetActive(p == page);
        shown = page;
        foreach (var (tab, p) in tabs)
        {
            tab.label.color = p == page ? Text : Muted;
            // The first page shown puts the underline there at once, later ones slide it over
            if (p != page || tabBar == null) continue;
            tabFrom = tabPlaced ? Mathf.Lerp(tabFrom, tabTo, Ease(tabT)) : tab.centre.x;
            tabTo = tab.centre.x;
            tabT = tabPlaced ? 0 : 1;
            tabPlaced = true;
        }
        Layout();
    }

    // Once, after the pages are built: a split page's columns to the panel's halves, the rest stays in the middle
    void Arrange(Page page)
    {
        if (page.arranged || page.splitChild < 0) return;
        page.arranged = true;
        for (int i = 0; i < page.root.childCount; i++)
            page.root.GetChild(i).localPosition += Vector3.right * (i < page.splitChild ? -Width / 2 : Width / 2);
        for (int i = 0; i < elements.Count; i++)
            if (elements[i].page == page) elements[i].centre.x += i < page.splitElement ? -Width / 2 : Width / 2;
    }

    void Layout()
    {
        height = -Mathf.Min(shown?.y ?? 0, shown != null && shown.splitChild >= 0 ? shown.firstEnd : 0) + Pad;
        if (back == null)
        {
            back = Shape(body, LBack, Vector3.zero, Vector2.one, Corner, new Color(Bg.r, Bg.g, Bg.b, 0.9f));
            backEdge = Shape(body, LBack, Vector3.zero, Vector2.one, Corner, LineStrong);
            topStrip = Shape(body, LEdge, Vector3.zero, Vector2.one, 0, Accent, true);
            var grip = new Element { kind = Kind.Handle, size = new Vector2(0.1f, 0.025f), name = "Move bar" };
            grip.face = Shape(root, LControl, Vector3.zero, new Vector2(0.08f, 0.008f), 0.004f, LineStrong);
            elements.Add(grip);
            handle = grip;
        }
        back.sharedMesh = RoundedRect(new Vector2(panelW, height), Corner, new Color(Bg.r, Bg.g, Bg.b, 0.92f));
        back.transform.localPosition = new Vector3(0, -height / 2, 0.003f);
        backEdge.sharedMesh = RoundedRect(new Vector2(panelW + 0.0016f, height + 0.0016f), Corner + 0.0008f, LineStrong);
        backEdge.transform.localPosition = new Vector3(0, -height / 2, 0.0035f);
        topStrip.sharedMesh = GradientBar(new Vector2(panelW - 2 * Corner, 0.0015f));
        topStrip.transform.localPosition = new Vector3(0, -0.0008f, 0.002f);
        body.localPosition = new Vector3(0, height / 2, 0);
        handle.centre = new Vector2(0, -height / 2 - 0.022f);
        handle.face.transform.localPosition = new Vector3(0, handle.centre.y, 0);
    }

    // Content

    public void Section(Page page, string title)
    {
        if (columnH > 0 && page.splitChild < 0 && page.y < page.top - columnH)
        {
            page.firstEnd = page.y;
            page.y = page.top;
            page.splitChild = page.root.childCount;
            page.splitElement = elements.Count;
        }
        page.sections++;
        page.y -= 0.012f;
        float y = page.y - 0.01f;
        var t = Label(page.root, new Vector3(-Width / 2 + Pad, y, 0), $"<color=#ff3d8b>{page.sections:00}</color>  /  {title.ToUpperInvariant()}",
            0.0105f, Muted, mono, TextAlignmentOptions.Left);
        t.characterSpacing = 12;
        sectionLabels.Add((page, t, $"{page.sections:00}  /  {title.ToUpperInvariant()}"));
        float start = -Width / 2 + Pad + t.preferredWidth + 0.01f, end = Width / 2 - Pad - 0.006f;
        for (float x = start; x < end - 0.004f; x += 0.006f)
            Shape(page.root, LEdge, new Vector3(x + 0.0015f, y, 0), new Vector2(0.003f, 0.0007f), 0, LineStrong);
        Shape(page.root, LEdge, new Vector3(end + 0.002f, y, 0), Vector2.one * 0.004f, 0.002f, Accent);
        page.y -= 0.03f;
    }

    public TextMeshPro Paragraph(Page page, string text, float size = 0.014f, bool muted = true, int lines = 1)
    {
        var t = Label(page.root, new Vector3(-Width / 2 + Pad, page.y, 0), text, size, muted ? Muted : Ui.Text, inter,
            TextAlignmentOptions.TopLeft);
        t.rectTransform.pivot = new Vector2(0, 1);
        t.rectTransform.sizeDelta = new Vector2(Width - 2 * Pad, lines * size * 1.5f);
        t.textWrappingMode = TextWrappingModes.Normal;
        page.y -= lines * size * 1.5f + 0.008f;
        return t;
    }

    // A big number with a caption under it, side by side with the others in the same call
    public TextMeshPro[] Readouts(Page page, params string[] captions)
    {
        var values = new TextMeshPro[captions.Length];
        float w = (Width - 2 * Pad) / captions.Length;
        for (int i = 0; i < captions.Length; i++)
        {
            float x = -Width / 2 + Pad + i * w;
            values[i] = Label(page.root, new Vector3(x, page.y - 0.016f, 0), "", 0.03f, Ui.Text, interBold, TextAlignmentOptions.Left);
            var c = Label(page.root, new Vector3(x, page.y - 0.042f, 0), captions[i].ToUpperInvariant(), 0.0095f, Muted, mono,
                TextAlignmentOptions.Left);
            c.characterSpacing = 12;
        }
        page.y -= 0.058f;
        return values;
    }

    public TextMeshPro Slider(Page page, string label, float min, float max, float step, Func<float> get, Action<float> set)
    {
        float y = page.y - RowH / 2;
        Label(page.root, new Vector3(-Width / 2 + Pad, y, 0), label, 0.015f, Ui.Text, inter, TextAlignmentOptions.Left);
        var value = Label(page.root, new Vector3(Width / 2 - Pad, y, 0), "", 0.013f, Ui.Text, mono, TextAlignmentOptions.Right);
        value.rectTransform.pivot = new Vector2(1, 0.5f);
        var e = new Element
        {
            kind = Kind.Slider, page = page, centre = new Vector2(0.045f, y), size = new Vector2(0.17f, 0.03f),
            get = get, set = set, min = min, max = max, step = step, name = label
        };
        e.face = Shape(page.root, LControl, new Vector3(e.centre.x, y, 0), new Vector2(e.size.x, 0.006f), 0.003f, Raised);
        e.fill = Shape(page.root, LFill, Vector3.zero, Vector2.one, 0.003f, Accent, true);
        e.knob = Shape(page.root, LKnob, Vector3.zero, Vector2.one * 0.016f, 0.008f, Ui.Text);
        e.label = value;
        elements.Add(e);
        page.y -= RowH;
        return value;
    }

    public void Toggle(Page page, string label, Func<bool> on, Action flip)
    {
        float y = page.y - RowH / 2;
        Label(page.root, new Vector3(-Width / 2 + Pad, y, 0), label, 0.015f, Ui.Text, inter, TextAlignmentOptions.Left);
        var e = new Element
        {
            kind = Kind.Toggle, page = page, centre = new Vector2(Width / 2 - Pad - 0.02f, y), size = new Vector2(0.04f, 0.022f),
            on = on, click = flip, name = label
        };
        e.face = Shape(page.root, LControl, new Vector3(e.centre.x, y, 0), e.size, 0.011f, Raised);
        e.fill = Shape(page.root, LFill, new Vector3(e.centre.x, y, -0.0005f), e.size, 0.011f, Accent, true);
        e.knob = Shape(page.root, LKnob, Vector3.zero, Vector2.one * 0.016f, 0.008f, Ui.Text);
        e.bar = HoldBar(page.root, new Vector3(-Width / 2 + Pad, y - RowH / 2 + 0.003f, 0));
        elements.Add(e);
        page.y -= RowH;
    }

    MeshFilter HoldBar(Transform parent, Vector3 pos)
    {
        var bar = Shape(parent, LText, pos, new Vector2(0.004f, 0.003f), 0.0015f, Accent, true);
        bar.gameObject.SetActive(false);
        return bar;
    }

    // Buttons across the panel; primary ones are white with black text, the rest outlined
    public void Buttons(Page page, params (string text, Action click)[] actions) => AddButtons(page, false, actions);

    public void Primary(Page page, string text, Action click) => AddButtons(page, true, (text, click));

    void AddButtons(Page page, bool primary, params (string text, Action click)[] actions)
    {
        float y = page.y - RowH / 2, gap = 0.008f;
        float w = (Width - 2 * Pad - (actions.Length - 1) * gap) / actions.Length;
        for (int i = 0; i < actions.Length; i++)
        {
            var e = new Element
            {
                kind = Kind.Button, page = page, primary = primary,
                centre = new Vector2(-Width / 2 + Pad + w / 2 + i * (w + gap), y), size = new Vector2(w, RowH * 0.82f),
                click = actions[i].click
            };
            var pos = new Vector3(e.centre.x, y, 0);
            e.edge = Shape(page.root, LControl, pos + new Vector3(0, 0, 0.0004f), e.size + Vector2.one * 0.0016f, 0.0038f, LineStrong);
            e.face = Shape(page.root, LFill, pos, e.size, 0.003f, primary ? Ui.Text : Bg);
            e.label = Label(page.root, pos, actions[i].text, 0.0135f, primary ? Color.black : Ui.Text, primary ? interBold : inter,
                TextAlignmentOptions.Center);
            e.label.rectTransform.sizeDelta = e.size;
            e.bar = HoldBar(page.root, pos + new Vector3(-e.size.x / 2, -e.size.y / 2 + 0.004f, 0));
            elements.Add(e);
        }
        page.y -= RowH;
    }

    public void Gap(Page page, float h = 0.008f) => page.y -= h;

    // A progress bar across the panel in the gradient; returns what sets it, 0 to 1
    public Action<float> Progress(Page page)
    {
        float y = page.y - 0.006f, w = Width - 2 * Pad, left = -Width / 2 + Pad;
        Shape(page.root, LControl, new Vector3(0, y, 0), new Vector2(w, 0.004f), 0.002f, Raised);
        var fill = Shape(page.root, LFill, Vector3.zero, Vector2.one, 0.002f, Accent, true);
        page.y -= 0.016f;
        return t =>
        {
            float fw = Mathf.Max(0.004f, w * Mathf.Clamp01(t));
            fill.sharedMesh = GradientBar(new Vector2(fw, 0.004f), 0.002f);
            fill.transform.localPosition = new Vector3(left + fw / 2, y, -0.0005f);
        };
    }

    // Buttons whose labels change (Pause / Continue); returns their labels
    public TextMeshPro[] ButtonsWithLabels(Page page, params (string text, Action click)[] actions)
    {
        int first = elements.Count;
        AddButtons(page, false, actions);
        var labels = new TextMeshPro[actions.Length];
        for (int i = 0; i < actions.Length; i++) labels[i] = elements[first + i].label;
        return labels;
    }

    public void Relayout() => Layout();

    // Placement

    // Opens within reach for a fingertip, a little below the eyes, tilted back towards the face
    public void Open(Transform head)
    {
        var forward = Vector3.ProjectOnPlane(head.forward, Vector3.up).normalized;
        if (forward.sqrMagnitude < 0.5f) forward = Vector3.forward;
        // A wide panel opens further out so both columns fit in view
        root.position = head.position + forward * (columnH > 0 ? 0.5f : 0.42f) + Vector3.down * 0.18f;
        root.rotation = Quaternion.LookRotation(forward, Vector3.up) * Quaternion.Euler(25, 0, 0);
        root.gameObject.SetActive(true);
        opened = 0;
    }

    public void Close() => root.gameObject.SetActive(false);

    // Interaction: a fingertip (poke) or a ray (a controller's, or the tracked hand pointing); down is the trigger or
    // the pinch for the ray, grab the fist or grip for the handle. A fingertip near the panel wins, out of reach the
    // ray takes over. Returns the ray's length to the panel, 0 when it misses
    // holdRay: a press through the ray has to be held (the bare hand's pinch); holdAll: every press does, and
    // holdStarted runs as soon as one begins (a recording pauses at the first touch)
    public float Interact(Vector3? tip, Ray? ray, bool down, bool grab, Transform grabber, out bool clicked,
        bool holdRay = false, bool holdAll = false, Action holdStarted = null)
    {
        clicked = false;
        if (!Visible)
        {
            hovered = null;
            HoverChanged = false;
            return 0;
        }
        Vector2 at = default;
        bool inFront = false, touching = false;
        float length = 0;
        bool poking = false;
        if (tip.HasValue)
        {
            var local = root.InverseTransformPoint(tip.Value);
            at = new Vector2(local.x, local.y - body.localPosition.y);
            inFront = local.z > -0.04f && local.z < 0.015f;
            poking = inFront || pressed != null && pressed != handle && wasTouching;
            touching = local.z > -0.004f && local.z < 0.015f;
            // A fingertip coming through from behind the panel doesn't press until it has left the surface
            if (touching && !wasTouching && !behind && prevZ > 0.015f) behind = true;
            if (!touching) behind = false;
            if (behind) touching = false;
            prevZ = local.z;
        }
        bool viaRay = false;
        if (!poking && ray.HasValue)
        {
            var plane = new Plane(root.forward, root.position);
            if (plane.Raycast(ray.Value, out float d) && d < 3)
            {
                var local = root.InverseTransformPoint(ray.Value.GetPoint(d));
                at = new Vector2(local.x, local.y - body.localPosition.y);
                inFront = true;
                touching = down;
                length = d;
                viaRay = true;
            }
        }
        Element hit = null;
        if (inFront)
            foreach (var e in elements)
            {
                if (e.page != null && e.page != shown) continue;
                var c = e.kind == Kind.Handle ? new Vector2(e.centre.x, e.centre.y - body.localPosition.y) : e.centre;
                var half = e.size / 2 + Vector2.one * 0.004f;
                if (Mathf.Abs(at.x - c.x) <= half.x && Mathf.Abs(at.y - c.y) <= half.y) hit = e;
            }
        releasedFor = touching ? 0 : releasedFor + Time.deltaTime;
        if (releasedFor >= ReleaseS) armed = true;
        // A press rides out a drop shorter than 0.1 s, the pinch flickering, instead of being let go
        bool still = touching || pressed != null && pressed != handle && releasedFor < 0.1f;
        if (armed && pressed == null && touching && hit != null && (!wasTouching || ray.HasValue) && hit.kind != Kind.Handle)
        {
            armed = false;
            pressed = hit;
            bool clickable = hit.kind == Kind.Button || hit.kind == Kind.Toggle;
            // Tabs only switch pages, they never need holding
            bool mustHold = clickable && hit.bar != null && (holdAll || holdRay && viaRay);
            hit.hold = 0;
            offFor = 0;
            hit.fired = !mustHold;
            if (mustHold) holdStarted?.Invoke();
            else if (clickable)
            {
                hit.click?.Invoke();
                clicked = true;
            }
        }
        if (pressed != null && !pressed.fired && pressed.kind != Kind.Handle)
        {
            // Sliding off the button or letting go before the time is up cancels it
            // Pinching swings the ray, so a held press keeps 1.5 cm of slack round its button and rides out leaving
            // it for under 0.15 s
            var pc = pressed.centre;
            var slack = pressed.size / 2 + Vector2.one * 0.015f;
            bool near = inFront && Mathf.Abs(at.x - pc.x) <= slack.x && Mathf.Abs(at.y - pc.y) <= slack.y;
            offFor = near ? 0 : offFor + Time.deltaTime;
            if (still && offFor < 0.15f)
            {
                pressed.hold += Time.deltaTime;
                if (pressed.hold >= HoldS)
                {
                    pressed.fired = true;
                    pressed.click?.Invoke();
                    clicked = true;
                }
            }
            else pressed.hold = 0;
        }
        if (grab && pressed == null && hit == handle && grabber != null)
        {
            pressed = handle;
            grabOffset = Quaternion.Inverse(grabber.rotation) * (root.position - grabber.position);
            grabTurn = Quaternion.Inverse(grabber.rotation) * root.rotation;
        }
        if (pressed == handle)
        {
            if (!grab) pressed = null;
            else
            {
                root.position = grabber.position + grabber.rotation * grabOffset;
                var turned = grabber.rotation * grabTurn;
                root.rotation = Quaternion.LookRotation(turned * Vector3.forward, Vector3.up);
            }
        }
        else if (pressed != null)
        {
            if (pressed.kind == Kind.Slider && touching)
            {
                float t = Mathf.InverseLerp(pressed.centre.x - pressed.size.x / 2, pressed.centre.x + pressed.size.x / 2, at.x);
                float v = Mathf.Lerp(pressed.min, pressed.max, t);
                pressed.set(Mathf.Round(v / pressed.step) * pressed.step);
            }
            if (!still) pressed = null;
        }
        wasTouching = touching;
        // A hand shaking at a control's edge drops off it for a frame or two: that isn't a new hover
        HoverChanged = hit != null && (hit != lastHovered || hoverOff > 0.3f);
        hoverOff = hit != null ? 0 : hoverOff + Time.deltaTime;
        if (hit != null) lastHovered = hit;
        hovered = hit;
        Draw(hit);
        // A ray anywhere on the panel counts, not only on a control, so its line doesn't blink off between buttons
        bool onPanel = viaRay && Mathf.Abs(at.x) <= panelW / 2 + 0.01f && at.y <= 0.01f && at.y >= -height - 0.04f;
        return hit != null || pressed != null || onPanel ? Mathf.Max(length, 0.0001f) : 0;
    }

    static float Ease(float t) => 1 - (1 - t) * (1 - t);

    static string NameOf(Element e) => e.name ?? (e.label != null ? e.label.text : "");

    bool OnShownPage(Element e) => e.page == null || e.page == shown;

    // The laptop points at a control on the page shown (or a tab, the close button, the move bar): it pulses for a
    // few seconds. False when nothing there has that name
    public bool Pulse(string page, string name)
    {
        if (!Visible || page != PageName) return false;
        var e = elements.Find(x => OnShownPage(x) && NameOf(x) == name);
        if (e == null) return false;
        e.pulse = PulseS;
        return true;
    }

    public void Describe(PanelOut o, string name)
    {
        o.name = name;
        o.page = PageName;
        o.w = panelW;
        o.h = height;
        var items = new List<ItemOut>();
        o.hover = o.press = -1;
        o.hold = 0;
        foreach (var (page, label, title) in sectionLabels)
            if (page == shown)
                items.Add(new ItemOut { n = title, k = "section", x = label.transform.localPosition.x, y = label.transform.localPosition.y });
        foreach (var e in elements)
        {
            if (!OnShownPage(e)) continue;
            var c = e.kind == Kind.Handle ? new Vector2(e.centre.x, e.centre.y - body.localPosition.y) : e.centre;
            string kind = e.kind == Kind.Toggle ? "toggle" : e.kind == Kind.Slider ? "slider" : e.kind == Kind.Handle ? "handle"
                : e == closeButton ? "close" : e.edge == null ? "tab" : e.primary ? "primary" : "button";
            var item = new ItemOut { n = NameOf(e), k = kind, x = c.x, y = c.y, w = e.size.x, h = e.size.y };
            // Toggles and sliders have their name at the left of their column
            if (e.kind == Kind.Toggle) item.lx = c.x + 0.02f + 2 * Pad - Width;
            if (e.kind == Kind.Slider)
            {
                item.lx = c.x - 0.045f - Width / 2 + Pad;
                item.v = Mathf.InverseLerp(e.min, e.max, e.get());
                item.t = e.label.text;
            }
            if (e.kind == Kind.Toggle) item.on = e.on();
            if (kind == "tab") item.on = tabs.Exists(t => t.tab == e && t.page == shown);
            if (e == hovered) o.hover = items.Count;
            if (e == pressed)
            {
                o.press = items.Count;
                o.hold = e.fired ? 0 : e.hold / HoldS;
            }
            items.Add(item);
        }
        o.items = items.ToArray();
    }

    // The panel's top centre and its axes in the world, scaled with it
    public Transform Body => body;

    void Draw(Element hit)
    {
        float dt = Time.deltaTime;
        opened = Mathf.MoveTowards(opened, 1, dt / OpenS);
        body.localScale = Vector3.one * Mathf.Lerp(0.92f, 1, Ease(opened));
        if (tabBar != null)
        {
            tabT = Mathf.MoveTowards(tabT, 1, dt / SlideS);
            tabBar.transform.localPosition = new Vector3(Mathf.Lerp(tabFrom, tabTo, Ease(tabT)), tabsY - 0.03f, -0.001f);
        }
        foreach (var e in elements)
        {
            if (e.page != null && e.page != shown) continue;
            e.hover = Mathf.MoveTowards(e.hover, e == hit ? 1 : 0, dt / 0.12f);
            // Pointed at from the laptop: lights up as on hover, on and off 1.5 times a second
            e.pulse = Mathf.MoveTowards(e.pulse, 0, dt);
            float glow = e.pulse > 0 ? 0.5f - 0.5f * Mathf.Cos((PulseS - e.pulse) * 3 * Mathf.PI) : 0;
            float lit = Mathf.Max(e.hover, glow);
            if (e.bar != null)
            {
                bool holding = e == pressed && !e.fired && e.hold > 0;
                e.bar.gameObject.SetActive(holding);
                if (holding)
                {
                    float w = Mathf.Max(0.004f, (e.kind == Kind.Toggle ? Width - 2 * Pad : e.size.x) * e.hold / HoldS);
                    e.bar.sharedMesh = GradientBar(new Vector2(w, 0.003f), 0.0015f);
                    var start = e.kind == Kind.Toggle ? e.centre.x + 0.02f + 2 * Pad - Width : e.centre.x - e.size.x / 2;
                    var y = e.kind == Kind.Toggle ? e.centre.y - RowH / 2 + 0.003f : e.centre.y - e.size.y / 2 + 0.004f;
                    e.bar.transform.localPosition = new Vector3(start + w / 2, y, -0.003f);
                }
            }
            e.press = Mathf.MoveTowards(e.press, e == pressed ? 1 : 0, dt / 0.08f);
            switch (e.kind)
            {
                case Kind.Button:
                    if (e.edge != null) Recolor(e.edge.sharedMesh, Color.Lerp(LineStrong, Accent, lit));
                    if (e.face != null)
                    {
                        var face = e.primary ? Color.Lerp(Ui.Text, Accent, Mathf.Max(e.press, glow) * 0.6f)
                            : Color.Lerp(Bg, Raised, lit);
                        Recolor(e.face.sharedMesh, Color.Lerp(face, Accent, e.primary ? 0 : e.press * 0.35f));
                        e.face.transform.localPosition = new Vector3(e.centre.x, e.centre.y, 0.002f * e.press);
                    }
                    if (e == closeButton) e.label.color = Color.Lerp(Muted, Accent, lit);
                    else if (e.edge == null && e.label != null)
                        e.label.color = Color.Lerp(tabs.Exists(t => t.tab == e && t.page == shown) ? Text : Muted, Accent, glow);
                    // Tabs (no box) keep the colour Show gives them
                    if (e.edge != null && e.label != null && !e.primary) e.label.color = Ui.Text;
                    break;
                case Kind.Toggle:
                {
                    float target = e.on() ? 1 : 0, was = e.slide;
                    e.slide = e.slide < 0 ? target : Mathf.MoveTowards(e.slide, target, dt / SlideS);
                    Recolor(e.face.sharedMesh, Color.Lerp(Raised, glow > e.hover ? Accent : LineStrong, lit));
                    float s = Ease(e.slide), travel = e.size.x / 2 - 0.011f;
                    float x = e.centre.x + Mathf.Lerp(-travel, travel, s);
                    e.knob.transform.localPosition = new Vector3(x, e.centre.y, -0.001f);
                    // The pink fill grows behind the knob from the left as it slides on
                    e.fill.gameObject.SetActive(s > 0.01f);
                    if (e.slide != was || was < 0)
                    {
                        float left = e.centre.x - e.size.x / 2, w = Mathf.Lerp(e.size.y, e.size.x, s);
                        e.fill.sharedMesh = GradientBar(new Vector2(w, e.size.y), e.size.y / 2);
                        e.fill.transform.localPosition = new Vector3(left + w / 2, e.centre.y, -0.0005f);
                    }
                    break;
                }
                case Kind.Slider:
                {
                    float t = Mathf.InverseLerp(e.min, e.max, e.get());
                    float left = e.centre.x - e.size.x / 2, w = Mathf.Max(0.006f, e.size.x * t);
                    e.fill.sharedMesh = GradientBar(new Vector2(w, 0.006f), 0.003f);
                    e.fill.transform.localPosition = new Vector3(left + w / 2, e.centre.y, -0.0005f);
                    e.knob.transform.localPosition = new Vector3(left + e.size.x * t, e.centre.y, -0.001f);
                    e.knob.transform.localScale = Vector3.one * (1 + 0.25f * lit);
                    break;
                }
                case Kind.Handle:
                    Recolor(e.face.sharedMesh, Color.Lerp(LineStrong, Accent, Mathf.Max(lit, e.press)));
                    break;
            }
        }
    }

    // Meshes

    TextMeshPro Label(Transform parent, Vector3 pos, string text, float size, Color color, TMP_FontAsset font,
        TextAlignmentOptions align)
    {
        var go = new GameObject("text");
        go.transform.SetParent(parent, false);
        var t = go.AddComponent<TextMeshPro>();
        t.font = font;
        t.text = text;
        // TextMeshPro's world text is about a tenth of a unit tall per point
        t.fontSize = size * 10;
        t.color = color;
        t.alignment = align;
        t.textWrappingMode = TextWrappingModes.NoWrap;
        t.rectTransform.sizeDelta = new Vector2(Width * 0.7f, RowH);
        t.rectTransform.pivot = new Vector2(align == TextAlignmentOptions.Center ? 0.5f : 0, 0.5f);
        t.sortingOrder = LText;
        go.transform.localPosition = pos + new Vector3(0, 0, -0.002f);
        return t;
    }

    MeshFilter Shape(Transform parent, int layer, Vector3 pos, Vector2 size, float corner, Color color, bool gradient = false)
    {
        var go = new GameObject("shape", typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetParent(parent, false);
        go.transform.localPosition = pos;
        var mf = go.GetComponent<MeshFilter>();
        mf.sharedMesh = gradient ? GradientBar(size, corner) : RoundedRect(size, corner, color);
        var r = go.GetComponent<MeshRenderer>();
        r.sharedMaterial = material;
        r.sortingOrder = layer;
        r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        r.receiveShadows = false;
        return mf;
    }

    static void Recolor(Mesh mesh, Color c)
    {
        var colors = new Color[mesh.vertexCount];
        for (int i = 0; i < colors.Length; i++) colors[i] = c;
        mesh.colors = colors;
    }

    // The site's gradient, orange to pink to magenta left to right
    public static Color GradientAt(float t) =>
        t < 0.5f ? Color.Lerp(Orange, Accent, t * 2) : Color.Lerp(Accent, Magenta, t * 2 - 1);

    static Mesh GradientBar(Vector2 size, float corner = 0)
    {
        var mesh = RoundedRect(size, Mathf.Min(corner, size.y / 2), Color.white);
        var v = mesh.vertices;
        var colors = new Color[v.Length];
        for (int i = 0; i < v.Length; i++) colors[i] = GradientAt(Mathf.InverseLerp(-size.x / 2, size.x / 2, v[i].x));
        mesh.colors = colors;
        return mesh;
    }

    // A flat rounded rectangle facing -z, as a triangle fan round its centre
    public static Mesh RoundedRect(Vector2 size, float r, Color color)
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
        var colors = new Color[v.Count];
        for (int i = 0; i < colors.Length; i++) colors[i] = color;
        mesh.colors = colors;
        return mesh;
    }

    // A label in the world (the cue card), same fonts as the menu
    public static TextMeshPro WorldLabel(Transform parent, Vector3 pos, string text, float size, Color color, bool bold,
        bool monoFont, TextAlignmentOptions align, Vector2 box)
    {
        var go = new GameObject("text");
        go.transform.SetParent(parent, false);
        var t = go.AddComponent<TextMeshPro>();
        t.font = monoFont ? mono : bold ? interBold : inter;
        t.text = text;
        t.fontSize = size * 10;
        t.color = color;
        t.alignment = align;
        t.textWrappingMode = TextWrappingModes.Normal;
        t.rectTransform.sizeDelta = box;
        t.sortingOrder = LText;
        go.transform.localPosition = pos;
        return t;
    }

    public static MeshFilter WorldShape(Transform parent, Material material, Vector3 pos, Vector2 size, float corner, Color color,
        int layer, bool gradient = false)
    {
        var go = new GameObject("shape", typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetParent(parent, false);
        go.transform.localPosition = pos;
        var mf = go.GetComponent<MeshFilter>();
        mf.sharedMesh = gradient ? GradientBar(size, corner) : RoundedRect(size, corner, color);
        var r = go.GetComponent<MeshRenderer>();
        r.sharedMaterial = material;
        r.sortingOrder = layer;
        r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        return mf;
    }

    public static void SetGradientBar(MeshFilter mf, Vector2 size, float corner) => mf.sharedMesh = GradientBar(size, corner);

    public static void SetColor(MeshFilter mf, Color c) => Recolor(mf.sharedMesh, c);
}
