using FaceMatchClient.DTOs;
using System.Diagnostics;
using System.Text.Json;

public static class FaceMatchService
{
    public static async Task<FaceMatchResponseDto> RunFaceMatchAsync(
        string helperExePath,
        string idCardImagePath,
        string cameraImagePath,
        int timeoutMs = 30_000) // 30s safety timeout
    {
        if (!File.Exists(helperExePath))
            throw new FileNotFoundException("FaceMatchHelper64.exe not found.", helperExePath);

        var psi = new ProcessStartInfo
        {
            FileName = helperExePath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden,
            WorkingDirectory = Path.GetDirectoryName(helperExePath) ?? Environment.CurrentDirectory
        };

        // full absolute paths are safer
        psi.ArgumentList.Add(Path.GetFullPath(idCardImagePath));
        psi.ArgumentList.Add(Path.GetFullPath(cameraImagePath));

        using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };

        process.Start();

        // read stdout & stderr concurrently to avoid deadlock
        var stdoutTask = process.StandardOutput.ReadToEndAsync();

        var stderrTask = process.StandardError.ReadToEndAsync();

        var exitTask = process.WaitForExitAsync();

        // optional timeout
        var timeoutTask = Task.Delay(timeoutMs);
        var finished = await Task.WhenAny(Task.WhenAll(stdoutTask, stderrTask, exitTask), timeoutTask);

        if (finished == timeoutTask)
        {
            try { if (!process.HasExited) process.Kill(true); } catch { /* ignore */ }
            throw new TimeoutException("FaceMatchHelper64 did not finish within timeout.");
        }

        string stdout = await stdoutTask;
        string stderr = await stderrTask;

        if (!string.IsNullOrWhiteSpace(stderr))
            Debug.WriteLine("FaceMatchHelper stderr: " + stderr);

        if (string.IsNullOrWhiteSpace(stdout))
            throw new Exception("FaceMatchHelper returned empty output. Stderr: " + stderr);

        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        FaceMatchResponseDto resp;
        try
        {
            resp = JsonSerializer.Deserialize<FaceMatchResponseDto>(stdout, options)
                   ?? throw new Exception("Deserialized response is null.");
        }
        catch (Exception ex)
        {
            throw new Exception(
                "Failed to parse FaceMatchHelper output: " + ex.Message + " | Raw: " + stdout);
        }

        return resp;
    }
}
