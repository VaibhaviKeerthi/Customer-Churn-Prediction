import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';
import { predictCustomerChurn } from './src/utils/churnModel';
import { Customer } from './src/types';

// Load environment variables
dotenv.config();

const app = express();
const PORT = 3000;

// Middleware for parsing JSON requests
app.use(express.json());

// Lazy initialize Gemini client to prevent crash if key is missing during container build/startup
let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn("WARNING: GEMINI_API_KEY is not defined. AI insights will operate in fallback mode.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey || "MOCK_KEY",
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

// ----------------------------------------------------
// API ROUTES
// ----------------------------------------------------

// 1. Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

// 2. Churn ML Prediction Endpoint (Local Algorithm)
app.post('/api/prediction/score', (req, res) => {
  try {
    const customer = req.body.customer as Customer;
    if (!customer) {
      res.status(400).json({ error: 'Missing customer dataset payload' });
      return;
    }
    const result = predictCustomerChurn(customer);
    res.json({ success: true, result });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// 3. AI Retention Playbook Generation
app.post('/api/prediction/explain', async (req, res) => {
  try {
    const { customer, scoringResult } = req.body;
    if (!customer || !scoringResult) {
      res.status(400).json({ error: 'Missing customer or scoringResult context' });
      return;
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
      // Return a professional mock response if API Key is not set up in secrets panel
      res.json({
        success: true,
        isMock: true,
        strategy: {
          customerName: customer.name,
          churnProbability: scoringResult.probability,
          riskDrivers: scoringResult.contributingFactors
            .filter((f: any) => f.impact === 'Positive')
            .map((f: any) => f.factor),
          recommendedPlay: customer.contract === 'Month-to-month' ? 'Annual Commitment Shift' : 'High-Value Bundle Incentives',
          winbackValue: `Safeguarding a customer with monthly billings of ₹${customer.monthlyCharges}. Historical tenure is ${customer.tenure} months.`,
          offerDetails: customer.contract === 'Month-to-month' 
            ? 'Transition to a secure 12-Month contract with a 15% rate discount for the first 4 months, plus complimentary Online Security.'
            : 'Unlocking a high-priority loyalty credit of ₹500, combined with a 6-month free trial of the Cyber Protection Suite.',
          playbookSteps: [
            "Proactively establish contact via account manager or priority outreach channel.",
            "Verify service performance issues, network outages, or device setup bugs.",
            "Highlight security features currently unattached, explaining their role in stabilizing connection safety.",
            "Offer direct billing convenience change (switch Paperless/Auto-Pay) & lock in rate discounts."
          ],
          estimatedSuccessProbability: Math.round(95 - (scoringResult.probability * 40))
        }
      });
      return;
    }

    const ai = getGeminiClient();
    const prompt = `Analyze this customer account for Churn Potential and formulate a hyper-personalized retention winback strategy:
    
    CUSTOMER DATA:
    Name: ${customer.name}
    Tenure: ${customer.tenure} months
    Contract: ${customer.contract}
    Internet Service: ${customer.internetService}
    Online Security: ${customer.onlineSecurity}
    Tech Support: ${customer.techSupport}
    Monthly Charges: ₹${customer.monthlyCharges}
    Payment Method: ${customer.paymentMethod}
    Senior Citizen: ${customer.seniorCitizen ? 'Yes' : 'No'}
    Partner/Dependents: ${customer.partner ? 'Has Partner' : 'No Partner'}, ${customer.dependents ? 'Has Dependents' : 'No Dependents'}
    
    SCORING ENGINE PREDICTION:
    Calculated Churn Probability: ${(scoringResult.probability * 100).toFixed(1)}%
    Core Vulnerabilities Detected: ${scoringResult.contributingFactors
      .filter((f: any) => f.impact === 'Positive')
      .map((f: any) => `${f.factor} (${f.description})`)
      .join('; ')}
    
    TASK:
    Develop an optimal retention playbook explaining what action steps an account executive should take, a specific discount/incentive offer, positive drivers to lean on, and an estimation of save success.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        systemInstruction: "You are an elite, highly professional retention strategist and data scientist. Provide clear, calculated corporate insights. Ensure all output maps precisely to the provided JSON structure. Note that prices and monetary benefits are described in Indian Rupees (₹).",
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            customerName: { type: Type.STRING },
            churnProbability: { type: Type.NUMBER },
            riskDrivers: {
              type: Type.ARRAY,
              items: { type: Type.STRING }
            },
            recommendedPlay: { type: Type.STRING, description: "Name of the target customer-save strategy" },
            winbackValue: { type: Type.STRING, description: "Why saving this specific customer makes financial and relationship sense" },
            offerDetails: { type: Type.STRING, description: "Specific price adjustment, bundle upgrade, or service attach credits" },
            playbookSteps: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Four sequential steps for an account representative to execute this recovery play"
            },
            estimatedSuccessProbability: { type: Type.INTEGER, description: "Dynamic save success likelihood percentage (0-100)" }
          },
          required: ["customerName", "churnProbability", "riskDrivers", "recommendedPlay", "winbackValue", "offerDetails", "playbookSteps", "estimatedSuccessProbability"]
        }
      }
    });

    const text = response.text;
    if (text) {
      const parsed = JSON.parse(text);
      res.json({ success: true, strategy: parsed });
    } else {
      throw new Error("Empty AI response received");
    }
  } catch (error: any) {
    console.error("Gemini Retention Blueprint Fail:", error);
    res.status(500).json({ error: error.message });
  }
});

// 4. Strategic Vulnerability & Macro Insights
app.post('/api/prediction/macro-insights', async (req, res) => {
  try {
    const { totalChurners, totalActive, highRiskCount, contractBreakdown, serviceBreakdown } = req.body;

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
      res.json({
        success: true,
        isMock: true,
        insights: "### Strategic Vulnerability Report (Local Analysis Mode)\n\n" +
          "**1. Core Concern: Month-to-Month Contracts**\n" +
          "The biggest driver of customer drop-off is Month-to-Month signups. Customers with no structural commitments account for over 80% of actual churners. Shifting these users towards loyalty contracts must be top organizational priority.\n\n" +
          "**2. Premium Fiber Pricing Exposures**\n" +
          "High average spend on Fiber Optic connections (₹799+/month) generates billing fatigue. When combined with zero security attachments, customers switch to budget alternatives quickly. Enrolling them in automated Auto-Pay decreases checking friction.\n\n" +
          "**3. Suggested Action Plan**\n" +
          "- *Mandatory Bundle Attachments:* Bundle free tech support trials with new DSL/Fiber signups.\n" +
          "- *Auto-Pay discount campaign:* Offering a small ₹50 monthly incentive to configure automatic billing pays for itself within three lifecycles."
      });
      return;
    }

    const ai = getGeminiClient();
    const prompt = `Analyze this global customer group metrics report and summarize the top core vulnerabilities, cohort exposures, and structural ways to prevent churn in India:
    
    METRICS SUMMARY:
    - Total Historical Churners: ${totalChurners} out of 40 active/inactive records
    - Active Users currently flagged as High-Risk: ${highRiskCount}
    - Contract distributions, Internet service breakouts, and billing ratios indicate month-to-month and unattached fiber plans are top issues. All monetary figures are in Indian Rupees (₹).
    
    TASK:
    Write a brief, high-impact consulting report with three main sections: Top Risk Exposures, Cohort Vulnerability Analysis, and Strategic Revenue Safeguards. Include specific tactical insights tailored to the Indian telecom market (where competitive pricing is high). Keep it business-focused.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        systemInstruction: "You are a senior telecommunication churn consultant. Write in a clear, executive tone. Use professional headings and bullet points."
      }
    });

    res.json({ success: true, insights: response.text });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// ----------------------------------------------------
// VITE OR STATIC FILE SERVING
// ----------------------------------------------------
async function initializeServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Customer Churn Server successfully listening on http://0.0.0.0:${PORT}`);
  });
}

initializeServer();
