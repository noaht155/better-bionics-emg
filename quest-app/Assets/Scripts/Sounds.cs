using UnityEngine;

// Short sounds for the menu, made at start so there are no audio files: a click when a press goes through, a softer
// tick when a hold starts and a faint one when the ray or a fingertip moves onto a control. Played where the head is,
// quiet.
public class Sounds
{
    public enum Kind { Click, Tick, Hover }

    const int Rate = 48000;
    readonly AudioClip click, tick, hover;

    public Sounds()
    {
        // A click: a quick rising blip, two tones a fifth apart. A tick: one higher, shorter, softer tone
        click = Tone("click", 0.045f, 0.35f, (t, d) => Mathf.Lerp(880, 1320, t / d));
        tick = Tone("tick", 0.02f, 0.18f, (t, d) => 1760);
        hover = Tone("hover", 0.012f, 0.06f, (t, d) => 2640);
    }

    public void Play(Kind kind, Vector3 at) =>
        AudioSource.PlayClipAtPoint(kind == Kind.Click ? click : kind == Kind.Tick ? tick : hover, at, 1);

    static AudioClip Tone(string name, float seconds, float volume, System.Func<float, float, float> hz)
    {
        int n = Mathf.RoundToInt(seconds * Rate);
        var data = new float[n];
        double phase = 0;
        for (int i = 0; i < n; i++)
        {
            float t = (float)i / Rate;
            phase += 2 * Mathf.PI * hz(t, seconds) / Rate;
            // 3 ms in, then dying away, so it doesn't pop
            float env = Mathf.Min(1, t / 0.003f) * Mathf.Exp(-t / (seconds / 3));
            data[i] = volume * env * (float)System.Math.Sin(phase);
        }
        var clip = AudioClip.Create(name, n, 1, Rate, false);
        clip.SetData(data, 0);
        return clip;
    }
}
