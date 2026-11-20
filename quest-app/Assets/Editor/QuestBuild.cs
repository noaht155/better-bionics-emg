using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEditor.XR.OpenXR.Features;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features.Interactions;
using UnityEngine.XR.OpenXR.Features.MetaQuestSupport;

// Sets the project up for the Quest 3 and builds the APK; build.sh runs it through the Unity CLI. Everything the
// build depends on is set here rather than by hand in the editor, so a fresh checkout builds the same.
public static class QuestBuild
{
    const string ScenePath = "Assets/Scenes/Main.unity";
    const string MaterialPath = "Assets/Materials/Sim.mat";
    const string Apk = "Build/bb-contact.apk";
    public const string Package = "com.betterbionics.contact";

    public static void Build()
    {
        Configure();
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { ScenePath }, locationPathName = Apk, target = BuildTarget.Android,
            options = BuildOptions.None
        });
        if (report.summary.result != BuildResult.Succeeded) EditorApplication.Exit(1);
    }

    [MenuItem("Quest/Configure")]
    public static void Configure()
    {
        EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);
        var android = NamedBuildTarget.Android;
        PlayerSettings.companyName = "Better Bionics";
        PlayerSettings.productName = "BB Contact";
        PlayerSettings.SetApplicationIdentifier(android, Package);
        PlayerSettings.SetScriptingBackend(android, ScriptingImplementation.IL2CPP);
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        // libmujoco and libbbsim are built for API 32
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
        PlayerSettings.Android.applicationEntry = AndroidApplicationEntry.Activity;
        // The link to the laptop is a plain websocket on localhost
        PlayerSettings.Android.forceInternetPermission = true;
        PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan });
        PlayerSettings.colorSpace = ColorSpace.Linear;
        ConfigureXR();
        ConfigurePlugins();
        MakeScene();
        AssetDatabase.SaveAssets();
    }

    static void ConfigureXR()
    {
        if (!EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey, out XRGeneralSettingsPerBuildTarget perTarget))
        {
            Directory.CreateDirectory("Assets/XR");
            perTarget = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
            AssetDatabase.CreateAsset(perTarget, "Assets/XR/XRGeneralSettingsPerBuildTarget.asset");
            EditorBuildSettings.AddConfigObject(XRGeneralSettings.k_SettingsKey, perTarget, true);
        }
        if (!perTarget.HasSettingsForBuildTarget(BuildTargetGroup.Android))
            perTarget.CreateDefaultSettingsForBuildTarget(BuildTargetGroup.Android);
        if (!perTarget.HasManagerSettingsForBuildTarget(BuildTargetGroup.Android))
            perTarget.CreateDefaultManagerSettingsForBuildTarget(BuildTargetGroup.Android);
        var settings = perTarget.SettingsForBuildTarget(BuildTargetGroup.Android);
        settings.InitManagerOnStart = true;
        XRPackageMetadataStore.AssignLoader(settings.AssignedSettings, "UnityEngine.XR.OpenXR.OpenXRLoader",
            BuildTargetGroup.Android);
        EditorUtility.SetDirty(settings);

        FeatureHelpers.RefreshFeatures(BuildTargetGroup.Android);
        var openxr = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android);
        if (openxr == null)
        {
            Debug.LogError("No OpenXR settings for Android");
            EditorApplication.Exit(1);
        }
        openxr.GetFeature<MetaQuestFeature>().enabled = true;
        openxr.GetFeature<OculusTouchControllerProfile>().enabled = true;
        // The panel's text shader isn't checked for single pass, and the scene is small enough to draw twice
        openxr.renderMode = OpenXRSettings.RenderMode.MultiPass;
        EditorUtility.SetDirty(openxr);
    }

    // build.sh copies libbbsim and libmujoco in; Unity has to be told they're arm64 Android libraries
    static void ConfigurePlugins()
    {
        AssetDatabase.Refresh();
        foreach (var path in Directory.GetFiles("Assets/Plugins/Android/arm64-v8a", "*.so"))
        {
            var importer = (PluginImporter)AssetImporter.GetAtPath(path.Replace('\\', '/'));
            importer.SetCompatibleWithAnyPlatform(false);
            importer.SetCompatibleWithEditor(false);
            importer.SetCompatibleWithPlatform(BuildTarget.Android, true);
            importer.SetPlatformData(BuildTarget.Android, "CPU", "ARM64");
            importer.SaveAndReimport();
        }
    }

    static void MakeScene()
    {
        Directory.CreateDirectory("Assets/Materials");
        Directory.CreateDirectory("Assets/Scenes");
        var material = AssetDatabase.LoadAssetAtPath<Material>(MaterialPath);
        if (material == null)
        {
            material = new Material(Shader.Find("Standard"));
            material.SetFloat("_Glossiness", 0.35f);
            AssetDatabase.CreateAsset(material, MaterialPath);
        }

        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene);
        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.35f, 0.35f, 0.38f);
        var sun = new GameObject("sun").AddComponent<Light>();
        sun.type = LightType.Directional;
        sun.intensity = 1.1f;
        sun.transform.rotation = Quaternion.Euler(55, -30, 0);

        var rig = new GameObject("rig");
        var head = new GameObject("head") { tag = "MainCamera" };
        head.transform.SetParent(rig.transform, false);
        var camera = head.AddComponent<Camera>();
        camera.nearClipPlane = 0.01f;
        camera.farClipPlane = 50;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = Color.black;

        var app = new GameObject("app").AddComponent<App>();
        app.material = material;
        app.head = head.transform;

        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
    }
}
