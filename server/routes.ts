import type { Express } from "express";
import type { Server } from "http";
import { storage, type AssessmentCreateInput } from "./storage";
import { requireAuth } from "./auth";
import { api } from "@shared/routes";
import { z } from "zod";
import { existsSync } from "fs";
import { writeFile, unlink } from "fs/promises";
import { randomUUID, createHash } from "crypto";
import { execFile } from "child_process";
import { promisify } from "util";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { rateLimit } from "express-rate-limit";

const execFileAsync = promisify(execFile);

/**
 * Tracks currently running inference requests to prevent
 * duplicate concurrent ML execution for identical payloads.
 */
const activeInferenceRequests = new Set<string>();

function generateRequestFingerprint(
  payload: unknown,
  userId: string,
): string {
  return createHash("sha256")
    .update(`${userId}::${JSON.stringify(payload)}`)
    .digest("hex");
}

// ESM-compatible path resolution for analyze.py
// Resolve relative to this source file, not process.cwd()
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const analyzePyPath = path.resolve(__dirname, "..", "analyze.py");

/**
 * Rate limiter for the ML assessment endpoint.
 * This endpoint spawns a Python subprocess for each request, which is resource-intensive.
 * Limits to 5 requests per minute per IP to prevent DoS attacks.
 */
const assessmentLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  limit: 5, // 5 requests per IP per window
  standardHeaders: "draft-8",
  legacyHeaders: false,
  message: {
    error: "Too many assessment requests. Please try again later.",
    retryAfter: 60, // seconds
  },
});

function getPythonExecutable() {
  const candidates = process.platform === "win32"
    ? [
        path.resolve(".venv", "Scripts", "python.exe"),
        path.resolve("venv", "Scripts", "python.exe")
      ]
    : [
        path.resolve(".venv", "bin", "python"),
        path.resolve("venv", "bin", "python")
      ];

  return candidates.find((candidate) => existsSync(candidate)) ?? "python3";
}

async function seedDatabase() {
  const existing = await storage.getAssessments();

  if (existing.length === 0) {
    console.log("Seeding database with sample assessments...");

    const samples: AssessmentCreateInput[] = [
      {
        gender: "Male",
        age: 45,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "never",
        bmi: "24.5",
        hba1cLevel: "5.2",
        bloodGlucoseLevel: "95",
        riskScore: "12.3",
        riskCategory: "LOW",
        factors: [
          { name: "Age", impact: "positive", description: "Increases risk" },
          { name: "Bmi", impact: "negative", description: "Lowers risk" },
          { name: "Hba1c Level", impact: "negative", description: "Lowers risk" }
        ],
        confidenceInterval: "8.5% - 16.1%",
        modelConfidence: "0.8770",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Female",
        age: 62,
        hypertension: true,
        heartDisease: false,
        smokingHistory: "former",
        bmi: "31.2",
        hba1cLevel: "6.8",
        bloodGlucoseLevel: "145",
        riskScore: "48.7",
        riskCategory: "MODERATE",
        factors: [
          { name: "Hba1c Level", impact: "positive", description: "Increases risk" },
          { name: "Bmi", impact: "positive", description: "Increases risk" },
          { name: "Hypertension", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "38.9% - 58.5%",
        modelConfidence: "0.5130",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Male",
        age: 58,
        hypertension: true,
        heartDisease: true,
        smokingHistory: "current",
        bmi: "35.8",
        hba1cLevel: "8.2",
        bloodGlucoseLevel: "198",
        riskScore: "76.4",
        riskCategory: "HIGH",
        factors: [
          { name: "Hba1c Level", impact: "positive", description: "Increases risk" },
          { name: "Blood Glucose Level", impact: "positive", description: "Increases risk" },
          { name: "Heart Disease", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "68.1% - 84.7%",
        modelConfidence: "0.7640",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Female",
        age: 22,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "never",
        bmi: "21.0",
        hba1cLevel: "4.8",
        bloodGlucoseLevel: "85",
        riskScore: "1.2",
        riskCategory: "LOW",
        factors: [
          { name: "Hba1c Level", impact: "negative", description: "Lowers risk" },
          { name: "Bmi", impact: "negative", description: "Lowers risk" }
        ],
        confidenceInterval: "0.1% - 2.3%",
        modelConfidence: "0.9880",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Male",
        age: 30,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "never",
        bmi: "23.5",
        hba1cLevel: "5.1",
        bloodGlucoseLevel: "90",
        riskScore: "2.1",
        riskCategory: "LOW",
        factors: [
          { name: "Hba1c Level", impact: "negative", description: "Lowers risk" }
        ],
        confidenceInterval: "0.5% - 3.7%",
        modelConfidence: "0.9790",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Female",
        age: 35,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "former",
        bmi: "22.0",
        hba1cLevel: "5.3",
        bloodGlucoseLevel: "92",
        riskScore: "3.4",
        riskCategory: "LOW",
        factors: [
          { name: "Hba1c Level", impact: "negative", description: "Lowers risk" }
        ],
        confidenceInterval: "1.1% - 5.7%",
        modelConfidence: "0.9660",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Male",
        age: 45,
        hypertension: true,
        heartDisease: false,
        smokingHistory: "former",
        bmi: "27.5",
        hba1cLevel: "5.9",
        bloodGlucoseLevel: "105",
        riskScore: "24.5",
        riskCategory: "MODERATE",
        factors: [
          { name: "Hypertension", impact: "positive", description: "Increases risk" },
          { name: "Bmi", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "16.1% - 32.9%",
        modelConfidence: "0.7550",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Female",
        age: 50,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "current",
        bmi: "29.0",
        hba1cLevel: "6.1",
        bloodGlucoseLevel: "110",
        riskScore: "31.2",
        riskCategory: "MODERATE",
        factors: [
          { name: "Bmi", impact: "positive", description: "Increases risk" },
          { name: "Hba1c Level", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "22.1% - 40.3%",
        modelConfidence: "0.6880",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Male",
        age: 40,
        hypertension: false,
        heartDisease: true,
        smokingHistory: "never",
        bmi: "26.2",
        hba1cLevel: "5.8",
        bloodGlucoseLevel: "102",
        riskScore: "28.7",
        riskCategory: "MODERATE",
        factors: [
          { name: "Heart Disease", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "19.8% - 37.6%",
        modelConfidence: "0.7130",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Female",
        age: 65,
        hypertension: true,
        heartDisease: true,
        smokingHistory: "never",
        bmi: "31.5",
        hba1cLevel: "7.2",
        bloodGlucoseLevel: "145",
        riskScore: "78.4",
        riskCategory: "HIGH",
        factors: [
          { name: "Hba1c Level", impact: "positive", description: "Increases risk" },
          { name: "Heart Disease", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "70.3% - 86.5%",
        modelConfidence: "0.7840",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Male",
        age: 72,
        hypertension: true,
        heartDisease: true,
        smokingHistory: "former",
        bmi: "33.0",
        hba1cLevel: "8.1",
        bloodGlucoseLevel: "180",
        riskScore: "92.1",
        riskCategory: "HIGH",
        factors: [
          { name: "Hba1c Level", impact: "positive", description: "Increases risk" },
          { name: "Age", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "86.8% - 97.4%",
        modelConfidence: "0.9210",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Male",
        age: 55,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "current",
        bmi: "35.5",
        hba1cLevel: "6.8",
        bloodGlucoseLevel: "135",
        riskScore: "65.3",
        riskCategory: "HIGH",
        factors: [
          { name: "Bmi", impact: "positive", description: "Increases risk" },
          { name: "Hba1c Level", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "56.0% - 74.6%",
        modelConfidence: "0.6530",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Female",
        age: 78,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "never",
        bmi: "20.5",
        hba1cLevel: "5.2",
        bloodGlucoseLevel: "88",
        riskScore: "12.4",
        riskCategory: "LOW",
        factors: [
          { name: "Age", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "8.6% - 16.2%",
        modelConfidence: "0.8760",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Female",
        age: 28,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "never",
        bmi: "38.2",
        hba1cLevel: "5.8",
        bloodGlucoseLevel: "115",
        riskScore: "22.1",
        riskCategory: "MODERATE",
        factors: [
          { name: "Bmi", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "13.9% - 30.3%",
        modelConfidence: "0.7790",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Male",
        age: 33,
        hypertension: true,
        heartDisease: false,
        smokingHistory: "current",
        bmi: "25.8",
        hba1cLevel: "5.6",
        bloodGlucoseLevel: "98",
        riskScore: "20.8",
        riskCategory: "MODERATE",
        factors: [
          { name: "Hypertension", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "12.8% - 28.8%",
        modelConfidence: "0.7920",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Male",
        age: 25,
        hypertension: false,
        heartDisease: false,
        smokingHistory: "never",
        bmi: "24.0",
        hba1cLevel: "11.5",
        bloodGlucoseLevel: "310",
        riskScore: "99.8",
        riskCategory: "HIGH",
        factors: [
          { name: "Hba1c Level", impact: "positive", description: "Increases risk" },
          { name: "Blood Glucose Level", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "99.4% - 100.0%",
        modelConfidence: "0.9980",
        createdBy: "drsmith@example.com"
      },
      {
        gender: "Female",
        age: 61,
        hypertension: true,
        heartDisease: true,
        smokingHistory: "former",
        bmi: "29.8",
        hba1cLevel: "6.5",
        bloodGlucoseLevel: "128",
        riskScore: "68.2",
        riskCategory: "HIGH",
        factors: [
          { name: "Hba1c Level", impact: "positive", description: "Increases risk" },
          { name: "Heart Disease", impact: "positive", description: "Increases risk" }
        ],
        confidenceInterval: "59.1% - 77.3%",
        modelConfidence: "0.6820",
        createdBy: "drsmith@example.com"
      }
    ];

    for (const sample of samples) {
      await storage.createAssessment(sample);
    }

    console.log("Seeding complete!");
  }
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // Seed database on startup — development only to prevent fake data in production
  if (process.env.NODE_ENV !== "production") {
    seedDatabase().catch(console.error);
  }

  app.post(
    api.assessments.preview.path,
    requireAuth,
    assessmentLimiter,
    async (req, res) => {
      try {
        const input = api.assessments.preview.input.parse(req.body);

        const tempFile = path.join(
          os.tmpdir(),
          `${randomUUID()}.json`
        );

        await writeFile(tempFile, JSON.stringify(input));

        try {
          const { stdout, stderr } = await execFileAsync(
            getPythonExecutable(),
            [analyzePyPath, "predict_file", tempFile],
            {
              timeout: 30000
            }
          );

          let prediction;

          try {
            prediction = JSON.parse(stdout.trim());
          } catch (e) {
            console.error(
              "Failed to parse python output (preview):",
              stdout,
              stderr
            );
            throw new Error("Failed to process prediction preview.");
          }

          if (prediction.error) {
            return res.status(400).json({
              message: prediction.error
            });
          }

          return res.json({
            riskScore: prediction.riskScore,
            riskCategory: prediction.riskCategory,
            factors: prediction.factors ?? [],
            confidenceInterval: prediction.confidenceInterval ?? null,
            modelConfidence: prediction.modelConfidence ?? null
          });
        } catch (error: any) {
          console.error("Python ML preview execution failed:", error);

          if (error.killed || error.signal === "SIGTERM") {
            return res.status(408).json({
              message: "Clinical assessment preview timed out."
            });
          }

          return res.status(500).json({
            message: "Failed to generate clinical preview."
          });
        } finally {
          try {
            await unlink(tempFile);
          } catch (e) {}
        }
      } catch (err) {
        if (err instanceof z.ZodError) {
          return res.status(400).json({
            message: err.errors[0].message
          });
        }

        console.error("Error creating assessment preview:", err);

        return res.status(500).json({
          message: "Internal server error"
        });
      }
    }
  );

  app.post(
    api.assessments.create.path,
    requireAuth,
    assessmentLimiter,
    async (req, res) => {
      let requestFingerprint: string | null = null;

      try {
        const input = api.assessments.create.input.parse(req.body);

        // Generate fingerprint for request deduplication
        const userId = req.session.user?.email;
        if (!userId) {
          return res.status(401).json({
            message: "Authentication required.",
          });
        }
        requestFingerprint = generateRequestFingerprint(input, userId);

        // Prevent duplicate concurrent inference execution
        if (activeInferenceRequests.has(requestFingerprint)) {
          return res.status(409).json({
            message:
              "An identical assessment request is already being processed."
          });
        }

        activeInferenceRequests.add(requestFingerprint);

        // Save input to a temporary file to pass to the Python script
        const tempFile = path.join(
          os.tmpdir(),
          `${randomUUID()}.json`
        );

        await writeFile(tempFile, JSON.stringify(input));

        try {
          // Call Python script to perform the logistic regression analysis
          const { stdout, stderr } = await execFileAsync(
            getPythonExecutable(),
            [analyzePyPath, "predict_file", tempFile],
            {
              timeout: 30000
            }
          );

          let prediction;

          try {
            prediction = JSON.parse(stdout.trim());

            if (prediction.error) {
              return res.status(400).json({
                message: prediction.error
              });
            }

          } catch (e) {
            console.error(
              "Failed to parse python output:",
              stdout,
              stderr
            );

            throw new Error("Failed to process prediction.");
          }

          // Ensure non-diagnostic framing in response
          prediction.disclaimer =
            "DISCLAIMER: This is a clinical decision support tool and is not a medical diagnosis. Please consult with a healthcare professional for clinical decisions.";

          // Save the assessment to the database
          const assessment = await storage.createAssessment({
            ...input,
            riskScore: String(prediction.riskScore),
            riskCategory: prediction.riskCategory,
            factors: prediction.factors,
            confidenceInterval: prediction.confidenceInterval,
            modelConfidence:
              prediction.modelConfidence == null
                ? undefined
                : String(prediction.modelConfidence),
            createdBy: userId
          });

          // Return both the DB assessment record and the rich prediction data
          res.status(201).json({
            ...assessment,
            prediction
          });

        } catch (error: any) {
          console.error("Python ML execution failed:", error);

          if (error.killed || error.signal === "SIGTERM") {
            return res.status(408).json({
              message: "Clinical assessment generation timed out."
            });
          }

          return res.status(500).json({
            message: "Failed to generate clinical assessment."
          });

        } finally {
          // Cleanup temporary file
          try {
            await unlink(tempFile);
          } catch (e) {}

          // Release active inference lock
          if (requestFingerprint) {
            activeInferenceRequests.delete(requestFingerprint);
          }
        }

      } catch (err) {
        // Release active inference lock on validation/runtime failure
        if (requestFingerprint) {
          activeInferenceRequests.delete(requestFingerprint);
        }

        if (err instanceof z.ZodError) {
          return res.status(400).json({
            message: err.errors[0].message
          });
        }

        console.error("Error creating assessment:", err);

        res.status(500).json({
          message: "Internal server error"
        });
      }
    }
  );

  app.get(api.assessments.list.path, requireAuth, async (req, res) => {
    try {
      const userEmail = req.session.user?.email;
      const assessments = await storage.getAssessments(50, 0, userEmail);

      res.json(assessments);

    } catch (err) {
      res.status(500).json({
        message: "Failed to fetch assessments"
      });
    }
  });

  return httpServer;
}
