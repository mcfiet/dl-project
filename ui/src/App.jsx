import { useMemo, useState } from "react";
import {
  AppBar,
  Box,
  Button,
  Card,
  Chip,
  Container,
  CssBaseline,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
  ThemeProvider,
  Toolbar,
  Typography,
  createTheme
} from "@mui/material";
import SportsMotorsportsIcon from "@mui/icons-material/SportsMotorsports";
import BoltIcon from "@mui/icons-material/Bolt";

const defaultForm = {
  driver_id: "hamilton",
  constructor_id: "mercedes",
  circuit_id: "melbourne",
  year: 2015,
  grid_position: 1,
  quali_delta: 0.0,
  quali_tm_delta: -0.59,
  season_pts_driver: 0.0,
  season_pts_team: 0.0,
  last_3_avg: 0.0,
  is_street_circuit: 1,
  is_wet: 0
};

const numberFields = new Set([
  "year",
  "grid_position",
  "quali_delta",
  "quali_tm_delta",
  "season_pts_driver",
  "season_pts_team",
  "last_3_avg",
  "is_street_circuit",
  "is_wet"
]);

const theme = createTheme({
  typography: {
    fontFamily: "\"Red Hat Display\", sans-serif",
    h3: { fontWeight: 700 },
    h6: { fontWeight: 600 }
  },
  palette: {
    mode: "light",
    primary: { main: "#e96b4b" },
    secondary: { main: "#4a8f8a" },
    background: {
      default: "#f6efe4",
      paper: "#fffaf2"
    }
  },
  shape: { borderRadius: 10 }
});

export default function App() {
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const statusLabel = useMemo(() => {
    if (!result) return "Bereit für eine Vorhersage.";
    return result.prediction === 1
      ? "Punkte wahrscheinlich."
      : "Punkte eher unwahrscheinlich.";
  }, [result]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: numberFields.has(name) ? value : value.trimStart()
    }));
  };

  const resetForm = () => {
    setForm(defaultForm);
    setResult(null);
    setError("");
  };

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    setResult(null);

    const payload = { ...form };
    for (const key of Object.keys(payload)) {
      if (numberFields.has(key)) {
        payload[key] = Number(payload[key]);
      }
    }

    try {
      const response = await fetch("https://dl-api.devoniq.de/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const detail = await response.json();
        throw new Error(detail.detail || "Vorhersage fehlgeschlagen");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: "100vh",
          background:
            "radial-gradient(circle at top left, rgba(249,239,224,1) 0%, rgba(243,230,211,1) 35%, rgba(230,243,241,1) 100%)"
        }}
      >
        <Container maxWidth="lg" sx={{ py: { xs: 4, md: 6 } }}>
          <Stack spacing={1} sx={{ mb: 4 }}>
            <Typography variant="h3">Vorhersage, ob ein Fahrer punktet</Typography>
            <Typography variant="body1" color="text.secondary">
              Trage die Renn-Features ein und lass das TabPFN-Inferenzmodell die
              Wahrscheinlichkeit berechnen.
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 760 }}>
              Das Modell basiert auf dem TabPFN-Ansatz und nutzt die gleichen
              Feature-Definitionen wie im Trainings-Notebook (Fahrer, Team, Strecke,
              Jahr sowie Rennkontext). Die Eingaben werden konsistent kodiert,
              anschließend liefert das Modell eine Wahrscheinlichkeit für
              Punkte im Rennen.
            </Typography>
          </Stack>

          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={3}
            alignItems="stretch"
          >
            <Card
              sx={{
                p: 3,
                flex: 1,
                minWidth: 280,
                display: "flex",
                flexDirection: "column",
                gap: 2,
                boxShadow: "0 24px 48px rgba(30,20,10,0.18)"
              }}
            >
              <Typography variant="h6">Status</Typography>
              <Typography color="text.secondary">
                {loading ? "Berechne..." : statusLabel}
              </Typography>
              <Stack direction="row" spacing={2}>
                <Card
                  variant="outlined"
                  sx={{ p: 2, flex: 1, bgcolor: "background.default" }}
                >
                  <Typography variant="caption" color="text.secondary">
                    Vorhersage
                  </Typography>
                  <Typography variant="h6" sx={{ mt: 0.5 }}>
                    {result ? (result.prediction ? "Punkte" : "Keine Punkte") : "—"}
                  </Typography>
                </Card>
                <Card
                  variant="outlined"
                  sx={{ p: 2, flex: 1, bgcolor: "background.default" }}
                >
                  <Typography variant="caption" color="text.secondary">
                    Wahrscheinlichkeit
                  </Typography>
                  <Typography variant="h6" sx={{ mt: 0.5 }}>
                    {result ? `${(result.probability * 100).toFixed(1)}%` : "—"}
                  </Typography>
                </Card>
              </Stack>
              <Divider />
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                <Typography variant="caption">
                  Backend: <code>/predict</code>
                </Typography>
                <Button variant="outlined" size="small" onClick={resetForm}>
                  Beispiel laden
                </Button>
              </Stack>
              <Card
                variant="outlined"
                sx={{
                  p: 2,
                  bgcolor: "#fff3e8",
                  borderStyle: "dashed",
                  borderColor: "rgba(233,107,75,0.5)"
                }}
              >
                <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                  Hinweis
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Verwende die gleichen IDs wie im Datensatz (z.B.{" "}
                  <code>hamilton</code>, <code>red_bull</code>,{" "}
                  <code>melbourne</code>). Das Modell kennt nur die Trainingsjahre.
                </Typography>
              </Card>
            </Card>

            <Card
              component="form"
              onSubmit={submit}
              sx={{
                p: 3,
                flex: 1.3,
                display: "grid",
                gap: 2,
                boxShadow: "0 24px 48px rgba(30,20,10,0.18)"
              }}
            >
              <Typography variant="h6">Feature Eingabe</Typography>
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                  gap: 2
                }}
              >
                <TextField
                  label="Fahrer-ID"
                  name="driver_id"
                  value={form.driver_id}
                  onChange={handleChange}
                  placeholder="hamilton"
                  required
                />
                <TextField
                  label="Team-ID"
                  name="constructor_id"
                  value={form.constructor_id}
                  onChange={handleChange}
                  placeholder="mercedes"
                  required
                />
                <TextField
                  label="Strecke-ID"
                  name="circuit_id"
                  value={form.circuit_id}
                  onChange={handleChange}
                  placeholder="melbourne"
                  required
                />
                <TextField
                  label="Saison"
                  name="year"
                  type="number"
                  value={form.year}
                  onChange={handleChange}
                  inputProps={{ min: 2015, max: 2025 }}
                  required
                />
                <TextField
                  label="Grid-Position"
                  name="grid_position"
                  type="number"
                  value={form.grid_position}
                  onChange={handleChange}
                  inputProps={{ min: 1, max: 22 }}
                  required
                />
                <TextField
                  label="Quali-Delta"
                  name="quali_delta"
                  type="number"
                  value={form.quali_delta}
                  onChange={handleChange}
                  inputProps={{ step: 0.001 }}
                  required
                />
                <TextField
                  label="Teamkollege-Delta"
                  name="quali_tm_delta"
                  type="number"
                  value={form.quali_tm_delta}
                  onChange={handleChange}
                  inputProps={{ step: 0.001 }}
                  required
                />
                <TextField
                  label="Saisonpunkte Fahrer"
                  name="season_pts_driver"
                  type="number"
                  value={form.season_pts_driver}
                  onChange={handleChange}
                  inputProps={{ step: 0.1 }}
                  required
                />
                <TextField
                  label="Saisonpunkte Team"
                  name="season_pts_team"
                  type="number"
                  value={form.season_pts_team}
                  onChange={handleChange}
                  inputProps={{ step: 0.1 }}
                  required
                />
                <TextField
                  label="Ø letzte 3 Rennen"
                  name="last_3_avg"
                  type="number"
                  value={form.last_3_avg}
                  onChange={handleChange}
                  inputProps={{ step: 0.1 }}
                  required
                />
                <FormControl>
                  <InputLabel id="street-circuit-label">Street Circuit</InputLabel>
                  <Select
                    labelId="street-circuit-label"
                    name="is_street_circuit"
                    label="Street Circuit"
                    value={form.is_street_circuit}
                    onChange={handleChange}
                  >
                    <MenuItem value={0}>Nein</MenuItem>
                    <MenuItem value={1}>Ja</MenuItem>
                  </Select>
                </FormControl>
                <FormControl>
                  <InputLabel id="wet-label">Nasses Rennen</InputLabel>
                  <Select
                    labelId="wet-label"
                    name="is_wet"
                    label="Nasses Rennen"
                    value={form.is_wet}
                    onChange={handleChange}
                  >
                    <MenuItem value={0}>Nein</MenuItem>
                    <MenuItem value={1}>Ja</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              {error ? (
                <Typography color="error">{error}</Typography>
              ) : null}

              <Button
                variant="contained"
                size="large"
                type="submit"
                disabled={loading}
              >
                {loading ? "Berechne..." : "Vorhersage starten"}
              </Button>
            </Card>
          </Stack>
        </Container>
      </Box>
    </ThemeProvider>
  );
}
