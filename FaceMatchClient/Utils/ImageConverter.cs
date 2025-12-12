using System;
using System.IO;


namespace FaceMatchClient.Utils
{


    public static class ImageConverter
    {

        public static Bitmap Base64ToBitmap(string base64)
        {
            if (string.IsNullOrWhiteSpace(base64))
                throw new ArgumentException("Base64 string is null or empty.", nameof(base64));

            byte[] bytes = Convert.FromBase64String(base64);

            using var ms = new MemoryStream(bytes);
            using var tmp = new Bitmap(ms);   // depends on stream

            // Clone to detach from the MemoryStream
            return new Bitmap(tmp);
        }
    }

}
