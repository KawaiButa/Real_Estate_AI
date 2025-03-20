# Real Estate Backend Application
## Introduction

This project is a backend application built using the Litestar Framework, designed to facilitate the posting of real estate information for buying and renting. The application provides a robust and scalable platform for users to manage real estate listings efficiently.

## Installation

To set up and run this application, follow these steps:

1. **Install Poetry**:
   If you haven't installed Poetry yet, you can do so by running the following command in your terminal:
```

curl -sSL https://install.python-poetry.org | sh

```

2. **Clone the Repository**:
Clone this repository to your local machine:
```

git clone https://github.com/your-repo-url/real-estate-backend.git

```

3. **Navigate to the Project Directory**:
```

cd real-estate-backend

```

4. **Install Dependencies**:
Use Poetry to install all required dependencies:
```

poetry install

```

5. **Create a PostgreSQL Database**:
Ensure you have PostgreSQL installed and create a new database for this application.

6. **Configure Environment Variables**:
Create a `.env` file in the root directory and add your database connection details:
```

SUPABASE_URL=<your_supabase_url>
SUPABASE_KEY=<your_supabase_key>
SMTP_SERVER=<your_mail_server>
SMTP_PORT=<your_mail_service_port>
SMTP_USERNAME=<your_email_username>
SMTP_PASSWORD=<your_email_password>
DB_URL=<your_DB_URL>

```

7. **Run Migrations**:
Run database migrations to set up the schema:
```

python -m litestar database upgrade

```

8. **Start the Application**:
Finally, start the Litestar application:
```
python -m litestar run

```

## Architecture

The application follows a layered architecture to ensure maintainability and scalability. Here's an overview of each layer:

### Controller
- **Purpose**: Handles incoming HTTP requests and sends responses.
- **Functionality**: Controllers act as the entry point for the application, receiving requests, calling services, and returning responses to the client.

### Service
- **Purpose**: Contains the business logic of the application.
- **Functionality**: Services interact with repositories to fetch or update data, perform complex operations, and validate inputs.

### Repository
- **Purpose**: Manages data access and storage.
- **Functionality**: Repositories encapsulate database operations, providing a layer of abstraction between services and the database.

### Database (PostgreSQL)
- **Purpose**: Stores all application data persistently.
- **Functionality**: PostgreSQL is used as the relational database management system to store real estate listings and related information.

This architecture ensures that each component has a clear responsibility, making the application easier to maintain and extend.
