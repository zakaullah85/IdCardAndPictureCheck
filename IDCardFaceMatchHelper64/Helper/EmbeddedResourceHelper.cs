using System;
using System.IO;
using System.Linq;
using System.Reflection;

namespace IDCardFaceMatchHelper64.Helper
{
  

    public static class EmbeddedResourceHelper
    {
        /// <summary>
        /// Extracts an embedded resource whose name ends with resourceFileName
        /// to a temp folder and returns its full path.
        /// The file will be overwritten if it already exists.
        /// </summary>
        public static string ExtractToTemp(string resourceFileName, string subFolderName = "FaceMatch")
        {
            var asm = Assembly.GetExecutingAssembly();
            var resourceName = asm
                .GetManifestResourceNames()
                .FirstOrDefault(n => n.EndsWith(resourceFileName, StringComparison.OrdinalIgnoreCase));

            if (resourceName == null)
                throw new InvalidOperationException($"Embedded resource not found: {resourceFileName}");

            string baseTemp = Path.Combine(Path.GetTempPath(), subFolderName);
            Directory.CreateDirectory(baseTemp);

            string targetPath = Path.Combine(baseTemp, resourceFileName);

            using (var stream = asm.GetManifestResourceStream(resourceName))
            {
                if (stream == null)
                    throw new InvalidOperationException($"Resource stream is null for: {resourceName}");

                using (var fs = new FileStream(targetPath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    stream.CopyTo(fs);
                }
            }

            return targetPath;
        }
    }

}
