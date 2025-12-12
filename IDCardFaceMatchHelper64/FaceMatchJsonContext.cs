using global::IDCardFaceMatchHelper64.DTOs;
using System.Text.Json.Serialization;


namespace IDCardFaceMatchHelper64
{
    [JsonSourceGenerationOptions(
        PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
        WriteIndented = false)]
    [JsonSerializable(typeof(FaceMatchResponse))]
    internal partial class FaceMatchJsonContext : JsonSerializerContext
    {
    }
}


