# Predicting F10.7 Solar Flux for the next 28 days
The following code provides a prediction for the next 28 days for F10.7cm flux density.

It does this by using historical F10.7cm data, alongside historical Sunspot number data and other flux densities F15cm and F8cm. Other drivers have been attempted, but only negatively detriment the model.

It makes use of an LSTM (Long Short-Term Memory) model that are well suited to time series  prediction tasks due to: its handling of temporal dependencies and its ability to capture patterns in time.