import { useEffect, useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Autocomplete,
  ToggleButton,
  ToggleButtonGroup,
  Slider,
  Alert,
  Chip,
  LinearProgress,
  Grid,
  Paper,
  Divider,
  Stack,
} from '@mui/material';
import {
  Male,
  Female,
  Search,
  Psychology,
  Public,
  Coronavirus,
  Air,
  Thermostat,
  WaterDrop,
  AccessTime,
  TrendingUp,
} from '@mui/icons-material';

const API_URL = import.meta.env.VITE_API_URL;
console.log('API_URL:', API_URL); // Debug: check if env is loaded

// Dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#6366f1',
    },
    secondary: {
      main: '#8b5cf6',
    },
    background: {
      default: '#0a0a0f',
      paper: '#12121a',
    },
  },
  typography: {
    fontFamily: '"Outfit", "Roboto", sans-serif',
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 16,
          border: '1px solid #2a2a3a',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
  },
});

interface DiseaseRisk {
  disease: string;
  disease_name: string;
  base_mortality_rate: number;
  adjusted_mortality_rate: number;
  risk_level: string;
  description: string;
}

interface PredictionResponse {
  country: string;
  gender: string;
  age: number;
  age_risk_factor: number;
  model_used: string;
  model_year_range: string;
  predictions: DiseaseRisk[];
  environmental_summary: {
    pm25: number;
    pm10: number;
    temperature_avg: number;
    humidity_avg: number;
    data_year: number;
  };
}

interface ModelInfo {
  name: string;
  description: string;
  year_range: string;
  n_features: number;
  overall_r2: number;
  features_type: string;
}

const RISK_COLORS: Record<string, string> = {
  'Very Low': '#22c55e',
  'Low': '#10b981',
  'Moderate': '#f59e0b',
  'High': '#f97316',
  'Very High': '#ef4444',
};

function App() {
  const [countries, setCountries] = useState<string[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('shortterm');
  const [gender, setGender] = useState<string>('male');
  const [age, setAge] = useState<number>(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/countries`)
      .then(res => res.json())
      .then(data => {
        setCountries(data.countries);
        if (data.countries.length > 0) {
          setSelectedCountry(data.countries[0]);
        }
      })
      .catch(() => setError('Failed to load countries. Is the server running?'));

    fetch(`${API_URL}/models`)
      .then(res => res.json())
      .then(data => setModels(data.models))
      .catch(() => console.log('Could not load model info'));
  }, []);

  const handlePredict = async () => {
    if (!selectedCountry) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/predict?model_type=${selectedModel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ country: selectedCountry, gender, age })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction failed');
      }

      setResult(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const currentModelInfo = models.find(m => m.name === selectedModel);

  const ageMarks = [
    { value: 1, label: '1' },
    { value: 25, label: '25' },
    { value: 50, label: '50' },
    { value: 75, label: '75' },
    { value: 100, label: '100' },
  ];

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* Header */}
        <Box sx={{ textAlign: 'center', py: 6, px: 3 }}>
          <Typography variant="h3" fontWeight={700} gutterBottom>
            🫁 Respiratory Risk Predictor
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Discover your respiratory disease risk based on where you live
          </Typography>
        </Box>

        <Container maxWidth="lg" sx={{ pb: 6 }}>
          <Grid container spacing={3}>
            {/* Input Panel */}
            <Grid size={{ xs: 12, md: 6 }}>
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Public /> Your Information
                  </Typography>
                  <Divider sx={{ mb: 3 }} />

                  {/* Country Selection */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      📍 {selectedCountry ?? 'country'}
                    </Typography>
                    <Autocomplete
                      value={selectedCountry}
                      onChange={(_, newValue) => setSelectedCountry(newValue)}
                      options={countries}
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          placeholder="Search countries..."
                          InputProps={{
                            ...params.InputProps,
                            startAdornment: <Search sx={{ color: 'text.secondary', mr: 1 }} />,
                          }}
                        />
                      )}
                      sx={{ '& .MuiOutlinedInput-root': { borderRadius: 3 } }}
                    />
                  </Box>

                  {/* Gender Selection */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      ⚧ Gender
                    </Typography>
                    <ToggleButtonGroup
                      value={gender}
                      exclusive
                      onChange={(_, value) => value && setGender(value)}
                      fullWidth
                      sx={{ '& .MuiToggleButton-root': { borderRadius: 3, py: 1.5 } }}
                    >
                      <ToggleButton value="male">
                        <Male sx={{ mr: 1 }} /> Male
                      </ToggleButton>
                      <ToggleButton value="female">
                        <Female sx={{ mr: 1 }} /> Female
                      </ToggleButton>
                    </ToggleButtonGroup>
                  </Box>

                  {/* Age Selection */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      🎂 Age: <Chip label={`${age} years`} size="small" color="primary" />
                    </Typography>
                    <Slider
                      value={age}
                      onChange={(_, value) => setAge(value as number)}
                      min={1}
                      max={100}
                      marks={ageMarks}
                      valueLabelDisplay="auto"
                      sx={{
                        '& .MuiSlider-track': {
                          background: 'linear-gradient(90deg, #22c55e, #f59e0b, #ef4444)',
                        },
                        '& .MuiSlider-rail': {
                          background: 'linear-gradient(90deg, #22c55e, #f59e0b, #ef4444)',
                          opacity: 0.3,
                        },
                      }}
                    />
                  </Box>

                  {/* Model Selection */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Psychology /> Analysis Model
                    </Typography>
                    <ToggleButtonGroup
                      value={selectedModel}
                      exclusive
                      onChange={(_, value) => value && setSelectedModel(value)}
                      fullWidth
                      sx={{ '& .MuiToggleButton-root': { borderRadius: 3, py: 1.5, flexDirection: 'column' } }}
                    >
                      <ToggleButton value="shortterm">
                        <Typography variant="body2" fontWeight={600}>Short-term</Typography>
                        <Typography variant="caption" color="text.secondary">2003-2023 • With Pollution</Typography>
                      </ToggleButton>
                      <ToggleButton value="longterm">
                        <Typography variant="body2" fontWeight={600}>Long-term</Typography>
                        <Typography variant="caption" color="text.secondary">1980-2023 • Climate Only</Typography>
                      </ToggleButton>
                    </ToggleButtonGroup>
                    {currentModelInfo && (
                      <Paper sx={{ mt: 1, p: 1.5, textAlign: 'center', bgcolor: 'background.default' }}>
                        <Typography variant="caption" color="text.secondary">
                          R² Score: <strong style={{ color: '#6366f1' }}>{(currentModelInfo.overall_r2 * 100).toFixed(1)}%</strong>
                          {' • '}{currentModelInfo.n_features} features
                        </Typography>
                      </Paper>
                    )}
                  </Box>

                  {/* Predict Button */}
                  <Button
                    variant="contained"
                    size="large"
                    fullWidth
                    onClick={handlePredict}
                    disabled={loading || !selectedCountry}
                    sx={{
                      py: 1.5,
                      background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
                      },
                    }}
                  >
                    {loading ? 'Analyzing...' : '🔮 Check My Risk'}
                  </Button>
                  {loading && <LinearProgress sx={{ mt: 2, borderRadius: 1 }} />}
                </CardContent>
              </Card>
            </Grid>

            {/* Results Panel */}
            <Grid size={{ xs: 12, md: 6 }}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TrendingUp /> Risk Assessment
                  </Typography>
                  <Divider sx={{ mb: 3 }} />

                  {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                      {error}
                    </Alert>
                  )}

                  {!result && !error && (
                    <Box sx={{ textAlign: 'center', py: 6 }}>
                      <Typography variant="h1" sx={{ opacity: 0.3, mb: 2 }}>🌍</Typography>
                      <Typography color="text.secondary" sx={{ mb: 3 }}>
                        Select your <strong>country</strong>, <strong>gender</strong>, and <strong>age</strong> to see your personalized risk assessment.
                      </Typography>
                      <Stack direction="row" spacing={2} justifyContent="center">
                        <Chip icon={<Coronavirus />} label="4 Diseases" variant="outlined" />
                        <Chip icon={<Public />} label={`${countries.length}+ Countries`} variant="outlined" />
                        <Chip icon={<Psychology />} label="2 AI Models" variant="outlined" />
                      </Stack>
                    </Box>
                  )}

                  {result && (
                    <Stack spacing={2}>
                      {/* User Summary */}
                      <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                          <Chip icon={<Public />} label={result.country} size="small" />
                          <Chip icon={result.gender === 'male' ? <Male /> : <Female />} label={result.gender === 'male' ? 'Male' : 'Female'} size="small" />
                          <Chip label={`${result.age} years`} size="small" />
                          <Chip label={`Age Factor: ${result.age_risk_factor}x`} size="small" color="primary" variant="outlined" />
                        </Stack>
                      </Paper>

                      {/* Model Used */}
                      <Alert severity="info" icon={<Psychology />}>
                        Model: <strong>{result.model_used === 'shortterm' ? 'Short-term' : 'Long-term'}</strong> ({result.model_year_range})
                      </Alert>

                      {/* Environmental Summary */}
                      <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                          <AccessTime fontSize="small" /> Environmental Data ({result.environmental_summary.data_year})
                        </Typography>
                        <Grid container spacing={2}>
                          {[
                            { icon: <Air />, label: 'PM2.5', value: `${result.environmental_summary.pm25} µg/m³` },
                            { icon: <Air />, label: 'PM10', value: `${result.environmental_summary.pm10} µg/m³` },
                            { icon: <Thermostat />, label: 'Temp', value: `${result.environmental_summary.temperature_avg}°C` },
                            { icon: <WaterDrop />, label: 'Humidity', value: `${result.environmental_summary.humidity_avg}%` },
                          ].map((item, i) => (
                            <Grid size={{ xs: 6, sm: 3 }} key={i}>
                              <Box sx={{ textAlign: 'center' }}>
                                {item.icon}
                                <Typography variant="caption" display="block" color="text.secondary">{item.label}</Typography>
                                <Typography variant="body2" fontWeight={600}>{item.value}</Typography>
                              </Box>
                            </Grid>
                          ))}
                        </Grid>
                      </Paper>

                      {/* Disease Cards */}
                      <Grid container spacing={2} sx={{ alignItems: 'stretch' }}>
                        {result.predictions.map((pred) => (
                          <Grid size={{ xs: 12, sm: 6 }} key={pred.disease} sx={{ display: 'flex' }}>
                            <Card
                              sx={{
                                borderLeft: `4px solid ${RISK_COLORS[pred.risk_level]}`,
                                transition: 'transform 0.2s',
                                '&:hover': { transform: 'translateY(-4px)' },
                                width: '100%',
                                display: 'flex',
                                flexDirection: 'column',
                              }}
                            >
                              <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 220 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                                  <Typography variant="subtitle2" sx={{ maxWidth: '60%' }}>{pred.disease_name}</Typography>
                                  <Chip
                                    label={pred.risk_level}
                                    size="small"
                                    sx={{ bgcolor: RISK_COLORS[pred.risk_level], color: 'white', fontWeight: 600 }}
                                  />
                                </Box>
                                <Typography variant="caption" color="text.secondary">
                                  Base: {pred.base_mortality_rate.toFixed(1)}
                                </Typography>
                                <Typography variant="h4" fontWeight={700} sx={{ my: 1 }}>
                                  {pred.adjusted_mortality_rate.toFixed(1)}
                                  <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                                    per 100k
                                  </Typography>
                                </Typography>
                                <LinearProgress
                                  variant="determinate"
                                  value={Math.min(100, (pred.adjusted_mortality_rate / 100) * 100)}
                                  sx={{
                                    height: 6,
                                    borderRadius: 3,
                                    bgcolor: 'background.default',
                                    '& .MuiLinearProgress-bar': { bgcolor: RISK_COLORS[pred.risk_level] },
                                  }}
                                />
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{
                                    mt: 'auto',
                                    pt: 1.5,
                                    display: 'block',
                                    lineHeight: 1.4,
                                  }}
                                >
                                  {pred.description}
                                </Typography>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Stack>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Container>


      </Box>
    </ThemeProvider>
  );
}

export default App;
