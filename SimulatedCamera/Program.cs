using System;
using System.IO;
using System.Threading;
using Microsoft.Data.SqlClient;
using System.Data;

class Program
{
    // Database connection string - CHANGE THESE VALUES TO MATCH YOUR SQL SERVER
    private static string connectionString = "Server=YOUR_SERVER;Database=YOUR_DATABASE;User Id=YOUR_USERNAME;Password=YOUR_PASSWORD;TrustServerCertificate=True;";

    static void Main(string[] args)
    {
        // Settings
        int pictureCount = 50; // Number of simulated pictures
        int intervalMinutes = 2; // Time interval in minutes

        // Output folder for simulated pictures
        string outputFolder = @"C:\Users\rdupart\OneDrive - Laborde Products Inc\camera\SimulatedImages";
        Directory.CreateDirectory(outputFolder);

        // Create database table if it doesn't exist
        CreateImageTableIfNotExists();

        Console.WriteLine("Simulated picture-taking started...");
        Console.WriteLine($"Saving simulated pictures to: {outputFolder}");

        // Loop to simulate picture-taking at intervals
        while (true)
        {
            Console.WriteLine($"Starting a new sequence of {pictureCount} pictures...");
            
            for (int i = 1; i <= pictureCount; i++)
            {
                // Simulate taking a picture
                string fileName = $"Picture_{DateTime.Now:yyyyMMdd_HHmmss}_{i}.jpg";
                string filePath = Path.Combine(outputFolder, fileName);

                // Create a dummy file to represent the picture
                File.WriteAllText(filePath, "Simulated picture content");

                // Save image information to database
                SaveImageToDatabase(fileName, filePath);

                Console.WriteLine($"Picture {i}/{pictureCount} saved as {fileName}");
                Thread.Sleep(500); // Simulate delay for taking each picture
            }

            Console.WriteLine($"Sequence complete. Waiting {intervalMinutes} minutes before the next sequence...");
            Thread.Sleep(intervalMinutes * 60 * 1000); // Wait before starting the next sequence
        }
    }

    private static void CreateImageTableIfNotExists()
    {
        try
        {
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                connection.Open();
                string createTableQuery = @"
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'SimulatedImages')
                    BEGIN
                        CREATE TABLE SimulatedImages (
                            ImageId INT IDENTITY(1,1) PRIMARY KEY,
                            FileName NVARCHAR(255) NOT NULL,
                            FilePath NVARCHAR(MAX) NOT NULL,
                            CreatedDate DATETIME NOT NULL,
                            Processed BIT DEFAULT 0,
                            ProcessedDate DATETIME NULL
                        )
                    END";

                using (SqlCommand command = new SqlCommand(createTableQuery, connection))
                {
                    command.ExecuteNonQuery();
                }
            }
            Console.WriteLine("Database table verified/created successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error creating database table: {ex.Message}");
            throw;
        }
    }

    private static void SaveImageToDatabase(string fileName, string filePath)
    {
        try
        {
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                connection.Open();
                string insertQuery = @"
                    INSERT INTO SimulatedImages (FileName, FilePath, CreatedDate)
                    VALUES (@FileName, @FilePath, @CreatedDate)";

                using (SqlCommand command = new SqlCommand(insertQuery, connection))
                {
                    command.Parameters.AddWithValue("@FileName", fileName);
                    command.Parameters.AddWithValue("@FilePath", filePath);
                    command.Parameters.AddWithValue("@CreatedDate", DateTime.Now);
                    command.ExecuteNonQuery();
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error saving to database: {ex.Message}");
        }
    }

    private static void ProcessSimulatedPictures(string folderPath)
    {
        Console.WriteLine("Processing simulated pictures...");

        var files = Directory.GetFiles(folderPath, "*.jpg");
        foreach (var file in files)
        {
            // Simulate reading the image (for now, just read the text content)
            string content = File.ReadAllText(file);

            // Simulate generating results
            Console.WriteLine($"Processing file: {Path.GetFileName(file)}");
            string result = $"Processed {Path.GetFileName(file)} at {DateTime.Now}";

            // Simulate saving results
            string resultFile = file.Replace(".jpg", "_result.txt");
            File.WriteAllText(resultFile, result);

            // Update database to mark image as processed
            UpdateImageProcessedStatus(Path.GetFileName(file));
        }

        Console.WriteLine("Processing complete.");
    }

    private static void UpdateImageProcessedStatus(string fileName)
    {
        try
        {
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                connection.Open();
                string updateQuery = @"
                    UPDATE SimulatedImages 
                    SET Processed = 1, ProcessedDate = @ProcessedDate
                    WHERE FileName = @FileName";

                using (SqlCommand command = new SqlCommand(updateQuery, connection))
                {
                    command.Parameters.AddWithValue("@FileName", fileName);
                    command.Parameters.AddWithValue("@ProcessedDate", DateTime.Now);
                    command.ExecuteNonQuery();
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error updating database: {ex.Message}");
        }
    }
}
