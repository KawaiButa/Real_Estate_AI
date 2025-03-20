# Real Estate Backend Application

## Introduction

This project is a backend application built using the [Litestar Framework](https://www.litestar.dev/). It is designed to facilitate the posting and management of real estate information for buying and renting. The application provides a robust and scalable platform for users to manage real estate listings efficiently.

## Installation

To set up and run this application, follow these steps:

1. **Install Poetry**  
   If you haven't installed Poetry yet, run the following command in your terminal:
   ```bash
   curl -sSL https://install.python-poetry.org | sh
   ```

2. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-repo-url/real-estate-backend.git
   ```

3. **Navigate to the Project Directory**  
   ```bash
   cd real-estate-backend
   ```

4. **Install Dependencies**  
   Use Poetry to install all required dependencies:
   ```bash
   poetry install
   ```

5. **Create a PostgreSQL Database**  
   Ensure PostgreSQL is installed on your system and create a new database for this application.

6. **Configure Environment Variables**  
   Create a `.env` file in the root directory and add your database connection details along with other environment variables:
   ```env
   SUPABASE_URL=<your_supabase_url>
   SUPABASE_KEY=<your_supabase_key>
   SMTP_SERVER=<your_mail_server>
   SMTP_PORT=<your_mail_service_port>
   SMTP_USERNAME=<your_email_username>
   SMTP_PASSWORD=<your_email_password>
   DB_URL=<your_DB_URL>
   ```

7. **Run Migrations**  
   Run database migrations to set up the schema:
   ```bash
   python -m litestar database upgrade
   ```

8. **Start the Application**  
   Finally, start the Litestar application:
   ```bash
   python -m litestar run
   ```

## Architecture

The application follows a layered architecture to ensure maintainability and scalability:

### Controller
- **Purpose:** Handles incoming HTTP requests and sends responses.
- **Functionality:** Acts as the entry point, receiving requests, calling services, and returning responses to the client.

### Service
- **Purpose:** Contains the business logic of the application.
- **Functionality:** Interacts with repositories to fetch or update data, perform complex operations, and validate inputs.

### Repository
- **Purpose:** Manages data access and storage.
- **Functionality:** Encapsulates database operations, providing a layer of abstraction between services and the database.

### Database (PostgreSQL)
- **Purpose:** Stores all application data persistently.
- **Functionality:** PostgreSQL is used as the relational database management system to store real estate listings and related information.

This clear separation of responsibilities makes the application easier to maintain and extend.

## Deployment

The project includes a **Dockerfile** for containerizing the application, making it simple to deploy on platforms such as Railwail.

### Deploy with Docker

1. **Build the Docker Image:**  
   Navigate to the project directory containing the Dockerfile and run:
   ```bash
   docker build -t real-estate-backend .
   ```

2. **Run the Container Locally:**  
   After building the image, run it locally to verify everything is working:
   ```bash
   docker run -d -p 8000:8000 real-estate-backend
   ```
   The application should now be accessible on `http://localhost:8000`.

3. **Deploy on Railwail (or other platforms):**  
   - Push your Docker image to a container registry (e.g., Docker Hub).
   - Configure your deployment platform (such as Railwail) to pull the image from the registry.
   - Set up environment variables, networking, and scaling options as needed.

## Known Issues

- **/api Route Error in Production:**  
  The `/api` route is currently returning a 500 Internal Server Error on the production server. This issue does not appear in the local environment. Work is in progress to resolve this problem.

## Live Demo

You can access the live application at:  
[https://real-estate-ai-287s.onrender.com](https://real-estate-ai-287s.onrender.com)
