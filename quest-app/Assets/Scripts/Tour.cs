using System;
using TMPro;
using UnityEngine;

// The first-start tour: one step at a time on a card behind the desk (where the cue card goes), each ticked off by
// what the user does, with the control it needs pulsing. Started on the first start, from Help, or from the
// laptop's VR tab.
public class Tour
{
    public class Step
    {
        public string text;
        public Func<bool> done;
        // Makes the control this step needs pulse (Ui.Pulse); null for none
        public Action point;
    }

    const float PointEveryS = 4, DoneShownS = 4;

    public readonly Ui card;
    readonly TextMeshPro count, body;
    readonly Step[] steps;
    int at = -1;
    float pointedAt, finishedAt = -1;

    public bool Running => at >= 0;

    public Tour(Material material, Step[] steps, Action<Ui> place)
    {
        this.steps = steps;
        card = new Ui(material, false);
        var page = card.AddPage("");
        card.Section(page, "Getting started");
        count = card.Paragraph(page, "", 0.011f, true);
        body = card.Paragraph(page, "", 0.016f, false, 4);
        card.Buttons(page, ("Skip this step", Next), ("End the tour", End));
        card.Show(page);
        place(card);
        card.Close();
    }

    public void Start()
    {
        at = -1;
        finishedAt = -1;
        card.root.gameObject.SetActive(true);
        Next();
    }

    public void End()
    {
        at = -1;
        finishedAt = -1;
        card.Close();
    }

    void Next()
    {
        at++;
        pointedAt = -PointEveryS;
        if (at < steps.Length) return;
        at = steps.Length;
        finishedAt = Time.time;
    }

    // Returns true when a step was just done, for a sound
    public bool Update()
    {
        if (!Running) return false;
        if (finishedAt >= 0)
        {
            count.text = "";
            body.text = "That's all. Help in the menu has it again, and the controls card at the desk's corner.";
            if (Time.time - finishedAt > DoneShownS) End();
            return false;
        }
        var step = steps[at];
        count.text = $"STEP {at + 1} / {steps.Length}";
        body.text = step.text;
        if (step.done())
        {
            Next();
            return true;
        }
        if (step.point != null && Time.time - pointedAt > PointEveryS)
        {
            pointedAt = Time.time;
            step.point();
        }
        return false;
    }
}
