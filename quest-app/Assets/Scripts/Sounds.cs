using UnityEngine;

// Short sounds for the menu, made at start so there are no audio files: a click when a press goes through and a
// softer tick when a hold starts. Played where the head is, quiet.
public class Sounds
{
    public enum Kind { Click, Tick }

    const int Rate = 48000;
    readonly AudioClip click, tick;

    public Sounds()
    {
        // A click: a quick rising blip, two tones a fifth apart. A tick: one higher, shorter, softer tone
        click = Tone("click", 0.045f, 0.35f, (t, d) => Mathf.Lerp(880, 1320, t / d));
        tick = Tone("tick", 0.02f, 0.18f, (t, d) => 1760);
    }

    public void Play(Kind kind, Vector3 at) => AudioSource.PlayClipAtPoint(kind == Kind.Click ? click : tick, at, 1);

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
