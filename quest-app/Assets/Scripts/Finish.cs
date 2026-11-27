using System;
using TMPro;
using UnityEngine;

// The end of a recording in the headset, as the Record tab's card shows it: the gradient ring drawing itself in, then
// a check (a cross when stopped early), the title, what was kept and the numbers. After a quick calibration it stays
// open with Calibrate on it and Control with it, so control can start without the laptop; otherwise a bar runs down
// and it closes itself after HoldS, as on the site.
public class Finish
{
    // As the site: ring 0.5 s, the mark from 0.4 s for 0.3 s, the text from 0.25 s, held 6 s
    const float RingS = 0.5f, MarkFrom = 0.4f, MarkS = 0.3f, TextFrom = 0.25f, TextS = 0.18f, HoldS = 6;
    // The ring's size and the room left for it at the top of the card (m, before the card's scale)
    const float MarkSize = 0.08f, MarkGap = 0.1f;
    // The site's 120 unit drawing: ring round (60, 60) with radius 54, the check and the cross's two strokes
    static readonly Vector2[] Check = { new Vector2(36, 62), new Vector2(53, 78), new Vector2(85, 44) };
    static readonly Vector2[] CrossA = { new Vector2(42, 42), new Vector2(78, 78) };
    static readonly Vector2[] CrossB = { new Vector2(78, 42), new Vector2(42, 78) };

    class Texts
    {
        public TextMeshPro meta, title, note, stats;
    }

    public readonly Ui ui;
    readonly Ui.Page quickPage, plainPage;
    readonly Texts quickTexts, plainTexts;
    readonly Action<float> countdown;
    readonly TextMeshPro calStatus;
    readonly TextMeshPro[] calButtons;
    readonly MeshFilter ring, strokeA, strokeB;
    readonly Vector3 markAt;
    Texts shown;
    bool quick, stopped, practice;
    float t;

    public bool Visible => ui.Visible;

    public Finish(Material material, Action calibrate, Action control)
    {
        ui = new Ui(material, false);
        quickPage = ui.AddPage("finish");
        quickTexts = Build(quickPage);
        calStatus = ui.Paragraph(quickPage, "", 0.012f, true, 2);
        calStatus.alignment = TextAlignmentOptions.Top;
        calButtons = ui.ButtonsWithLabels(quickPage, ("Calibrate on it", calibrate), ("Control with it", control));
        ui.Buttons(quickPage, ("Close", Close));
        plainPage = ui.AddPage("finish");
        plainTexts = Build(plainPage);
        countdown = ui.Progress(plainPage);
        ui.Buttons(plainPage, ("Close", Close));
        // Both pages start at the same height, so the mark sits once on the panel's body over the gap Build leaves
        markAt = new Vector3(0, quickPage.top - MarkGap / 2, -0.002f);
        ring = Ui.WorldShape(ui.Body, material, markAt, Vector2.one, 0, Color.white, 5);
        strokeA = Ui.WorldShape(ui.Body, material, markAt, Vector2.one, 0, Color.white, 5);
        strokeB = Ui.WorldShape(ui.Body, material, markAt, Vector2.one, 0, Color.white, 5);
        ui.Show(quickPage);
        ui.Close();
    }

    Texts Build(Ui.Page page)
    {
        ui.Gap(page, MarkGap);
        var texts = new Texts
        {
            meta = ui.Paragraph(page, "", 0.011f, true),
            title = ui.Paragraph(page, "", 0.026f, false),
            note = ui.Paragraph(page, "", 0.013f, true, 2),
            stats = ui.Paragraph(page, "", 0.013f, false)
        };
        foreach (var x in new[] { texts.meta, texts.title, texts.note, texts.stats }) x.alignment = TextAlignmentOptions.Top;
        return texts;
    }

    // Site units (0 to 120, y down) to metres round the mark's centre
    static Vector3[] At(Vector2[] u) =>
        Array.ConvertAll(u, p => new Vector3((p.x - 60) / 120 * MarkSize, -(p.y - 60) / 120 * MarkSize, 0));

    // At the cue card's place (place: the card's transform, same parent)
    public void Show(Link.Last last, Transform place)
    {
        quick = last.quick && last.completed && !last.practice;
        stopped = !last.completed;
        practice = last.practice && last.completed;
        shown = quick ? quickTexts : plainTexts;
        ui.Show(quick ? quickPage : plainPage);
        ui.root.SetParent(place.parent, false);
        ui.root.SetLocalPositionAndRotation(place.localPosition, place.localRotation);
        ui.root.localScale = place.localScale;
        shown.meta.text = (practice || last.practice ? "PRACTICE RUN" : quick ? "QUICK CALIBRATION" : "RECORDING") + "   ·   "
                          + (last.completed ? $"CUE {last.cues} / {last.cues}" : $"STOPPED AT CUE {last.cues_done + 1} / {last.cues}");
        shown.title.text = stopped ? "Stopped early" : practice ? "Practice run done" : "Session complete";
        string marked = last.marked_bad > 0
            ? $" <color=#ffc233>{last.marked_bad} cue{(last.marked_bad == 1 ? "" : "s")} marked bad, left out of training.</color>"
            : "";
        shown.note.text = stopped ? last.practice ? "Nothing was saved." : "Saved, not used for training because it didn't finish."
            : last.practice ? "Nothing was saved." + marked
            : last.hum_passed ? "Saved. It had hum, so it only calibrates and is left out of training." + marked
            : quick ? "Saved. Calibrate the network on it, then control with it." + marked
            : "Saved, ready for training." + marked;
        int s = Mathf.RoundToInt(last.seconds);
        shown.stats.text = $"{s / 60}:{s % 60:00} <size=70%><color=#a1a1aa>DURATION</color></size>"
                           + (last.marked_bad > 0 ? $"     {last.marked_bad} <size=70%><color=#a1a1aa>MARKED BAD</color></size>" : "")
                           + (last.hum_pauses > 0 ? $"     {last.hum_pauses} <size=70%><color=#a1a1aa>HUM REDOS</color></size>" : "");
        t = 0;
        Draw();
        ui.root.gameObject.SetActive(true);
    }

    public void Close() => ui.Close();

    // train: the calibration the headset started on this session (Link.Train), for the quick card
    public void Update(Link.Train train)
    {
        if (!Visible) return;
        t += Time.deltaTime;
        if (t < MarkFrom + MarkS + 0.1f) Draw();
        float a = Mathf.Clamp01((t - TextFrom) / TextS);
        shown.meta.alpha = shown.title.alpha = a;
        shown.note.alpha = Mathf.Clamp01((t - TextFrom - 0.07f) / TextS);
        shown.stats.alpha = Mathf.Clamp01((t - TextFrom - 0.15f) / TextS);
        if (quick)
        {
            bool saved = train != null && !string.IsNullOrEmpty(train.saved);
            calStatus.text = train == null ? "Laptop not connected"
                : !string.IsNullOrEmpty(train.error) ? $"<color=#ff3b3b>{train.error}</color>"
                : train.running || saved ? train.status
                : "Fits the network to where the band sits now, then scores it.";
            calButtons[0].text = train != null && train.running ? "Calibrating..." : saved ? "Calibrate again" : "Calibrate on it";
            return;
        }
        countdown(1 - t / HoldS);
        if (t >= HoldS) Close();
    }

    void Draw()
    {
        float r = MarkSize * 54 / 120, width = MarkSize * 6 / 120;
        Intro.Replace(ring, Intro.Arc(Intro.Ease(Mathf.Clamp01(t / RingS)), r, width, MarkSize));
        if (stopped) Ui.SetColor(ring, Ui.Muted);
        else if (practice) Ui.SetColor(ring, Ui.Text);
        var color = stopped ? Ui.Muted : Ui.Text;
        float m = Mathf.Clamp01((t - MarkFrom) / MarkS);
        if (stopped)
        {
            Intro.Replace(strokeA, Intro.Polyline(At(CrossA), Intro.Ease(Mathf.Clamp01(2 * m)), width, color));
            Intro.Replace(strokeB, Intro.Polyline(At(CrossB), Intro.Ease(Mathf.Clamp01(2 * m - 1)), width, color));
        }
        else
        {
            Intro.Replace(strokeA, Intro.Polyline(At(Check), Intro.Ease(m), width, color));
            Intro.Replace(strokeB, new Mesh());
        }
    }
}
