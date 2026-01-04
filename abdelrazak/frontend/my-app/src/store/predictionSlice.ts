import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ;

// Types
export interface EnvironmentalInput {
  pm25: number;
  pm10: number;
  no2: number;
  o3: number;
  so2: number;
  temp: number;
  humidity: number;
  precip: number;
  wind: number;
}

export interface PredictionResult {
  disease: string;
  disease_name: string;
  mortality_rate: number;
  risk_level: string;
}

interface PredictionState {
  input: EnvironmentalInput;
  predictions: PredictionResult[];
  selectedDisease: string | null;
  loading: boolean;
  error: string | null;
  availableDiseases: Record<string, string>;
}

const initialInput: EnvironmentalInput = {
  pm25: 25,
  pm10: 50,
  no2: 20,
  o3: 60,
  so2: 10,
  temp: 15,
  humidity: 60,
  precip: 800,
  wind: 3,
};

const initialState: PredictionState = {
  input: initialInput,
  predictions: [],
  selectedDisease: null,
  loading: false,
  error: null,
  availableDiseases: {},
};

// Async Thunks
export const fetchDiseases = createAsyncThunk('prediction/fetchDiseases', async () => {
  const response = await fetch(`${API_BASE_URL}/diseases`);
  if (!response.ok) throw new Error('Failed to fetch diseases');
  const data = await response.json();
  return data.diseases as Record<string, string>;
});

export const predictAll = createAsyncThunk(
  'prediction/predictAll',
  async (input: EnvironmentalInput) => {
    const response = await fetch(`${API_BASE_URL}/predict/all`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    });
    if (!response.ok) throw new Error('Prediction failed');
    const data = await response.json();
    return data.predictions as PredictionResult[];
  }
);

export const predictSingle = createAsyncThunk(
  'prediction/predictSingle',
  async ({ disease, input }: { disease: string; input: EnvironmentalInput }) => {
    const response = await fetch(`${API_BASE_URL}/predict/${disease}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    });
    if (!response.ok) throw new Error('Prediction failed');
    return (await response.json()) as PredictionResult;
  }
);

// Slice
const predictionSlice = createSlice({
  name: 'prediction',
  initialState,
  reducers: {
    updateInput: (state, action: PayloadAction<Partial<EnvironmentalInput>>) => {
      state.input = { ...state.input, ...action.payload };
    },
    setSelectedDisease: (state, action: PayloadAction<string | null>) => {
      state.selectedDisease = action.payload;
    },
    resetInput: (state) => {
      state.input = initialInput;
    },
    clearPredictions: (state) => {
      state.predictions = [];
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDiseases.fulfilled, (state, action) => {
        state.availableDiseases = action.payload;
      })
      .addCase(predictAll.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(predictAll.fulfilled, (state, action) => {
        state.loading = false;
        state.predictions = action.payload;
      })
      .addCase(predictAll.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Prediction failed';
      })
      .addCase(predictSingle.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(predictSingle.fulfilled, (state, action) => {
        state.loading = false;
        state.predictions = [action.payload];
      })
      .addCase(predictSingle.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Prediction failed';
      });
  },
});

export const { updateInput, setSelectedDisease, resetInput, clearPredictions } =
  predictionSlice.actions;
export default predictionSlice.reducer;

