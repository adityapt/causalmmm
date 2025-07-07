# causalmmm
A repository to build causal MMM models

## Overview
This is a Streamlit web application called CausalMMM (Causal Marketing Mix Modeling) built for marketing analytics. The application allows users to upload marketing data, select variables dynamically, perform causal analysis with GRU + Bayesian Networks, and visualize results in real-time. It uses Python with Streamlit frontend and PyTorch ML backend.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with Python
- **Visualization**: Plotly for interactive charts and graphs
- **UI Components**: Streamlit native components with custom CSS
- **Data Processing**: Pandas, NumPy for data manipulation
- **Real-time Updates**: Streamlit's reactive programming model

### ML Backend Architecture
- **Framework**: PyTorch for deep learning models
- **Model**: Causal GRU MMM with Bayesian Networks
- **Data Processing**: Scikit-learn for preprocessing and metrics
- **Visualization**: Matplotlib and Plotly for model outputs

### Data Flow
The application uses session state for data persistence:
1. **raw_data**: Uploaded CSV data
2. **variable_config**: User-selected variable mappings
3. **training_results**: Complete model outputs and metrics

## Key Components

### Data Management
- **Storage Interface**: Abstracted storage layer with in-memory implementation for development
- **CSV Upload**: Handles bulk import of marketing data via CSV files
- **Data Validation**: Uses Drizzle-Zod for schema validation

### Analytics Features
- **Causal Analysis**: Performs statistical analysis on marketing data
- **Visualization**: Interactive charts using Plotly.js
- **Metrics Dashboard**: Real-time calculation of ROAS, conversion rates, and spending metrics
- **Export Functionality**: Export data and results in CSV/JSON formats

### User Interface
- **Dashboard**: Main interface with sidebar navigation
- **Data Upload**: Drag-and-drop CSV file upload
- **Data Preview**: Table view of uploaded marketing data
- **Analysis Metrics**: Cards showing key performance indicators
- **Visualization Section**: Interactive charts and graphs

## Data Flow

1. **Data Upload**: Users upload CSV files containing marketing data
2. **Data Processing**: CSV files are parsed and validated against schema
3. **Data Storage**: Validated data is stored in PostgreSQL database
4. **Analysis**: Causal analysis algorithms process the stored data
5. **Visualization**: Results are displayed through interactive charts and metrics
6. **Export**: Users can export processed data and analysis results

## External Dependencies

### Core Dependencies
- **@neondatabase/serverless**: PostgreSQL database connection
- **drizzle-orm**: Database ORM and query builder
- **@tanstack/react-query**: Server state management
- **csv-parser**: CSV file processing
- **plotly.js**: Data visualization library
- **lucide-react**: Icon library

### Development Tools
- **Vite**: Build tool and development server
- **TypeScript**: Type safety and development experience
- **TailwindCSS**: Utility-first CSS framework
- **@replit/vite-plugin-runtime-error-modal**: Development error handling

## Deployment Strategy

### Development Environment
- **Frontend**: Vite development server on port 5173
- **Backend**: Express server on port 3000
- **Database**: Neon PostgreSQL connection via environment variables

### Production Build
- **Frontend**: Vite builds static assets to `dist/public`
- **Backend**: esbuild bundles server code to `dist/index.js`
- **Database**: Drizzle migrations applied via `db:push` command

### Environment Configuration
- **DATABASE_URL**: PostgreSQL connection string (required)
- **NODE_ENV**: Environment mode (development/production)
- **PORT**: Server port (defaults to 3000)
Frontend: Vite builds static assets to dist/public
Backend: esbuild bundles server code to dist/index.js
Database: Drizzle migrations applied via db:push command
Environment Configuration
DATABASE_URL: PostgreSQL connection string (required)
NODE_ENV: Environment mode (development/production)
PORT: Server port (defaults to 3000)
