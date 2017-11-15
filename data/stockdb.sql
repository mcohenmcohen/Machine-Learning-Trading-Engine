--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner:
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;


CREATE TABLE symbols (
    id serial primary key,
    Symbol character varying NOT NULL,
    Date timestamp NOT NULL,
    Open real,
    Low real,
    High real,
    Close real,
    Volume real,
    unique (Symbol, Date)

	--Volume integer
	-- ,PRIMARY KEY (symbol, date)
);

--CREATE TYPE model AS ENUM ('Random Forest Classifier', 'ok', 'happy');
CREATE TABLE models (
    id serial primary key,
    Name character varying NOT NULL,
    Abbrev character varying NOT NULL,
    unique (Name, Abbrev)
);
insert into models (name, abbrev) values ('Random Forest Classifier', 'rfc');
insert into models (name, abbrev) values ('Radnom Forest Regressor','rfr');
insert into models (name, abbrev) values ('AdaBoost Classifier','adc');
insert into models (name, abbrev) values ('AdaBoost Regressor','adr');
insert into models (name, abbrev) values ('Gradient Boost Classifier','gbc');
insert into models (name, abbrev) values ('Gradient boost Regressor','gbr');
insert into models (name, abbrev) values ('K Nearest Neghbors','knn');
insert into models (name, abbrev) values ('Support Vector Machine Classifier','svc');
insert into models (name, abbrev) values ('Support Vector Machine Regressor','svr');
insert into models (name, abbrev) values ('Linear Regression','LR');
insert into models (name, abbrev) values ('Logistic Regression','LogR');
insert into models (name, abbrev) values ('Ridge Regression','Ridge');
insert into models (name, abbrev) values ('Lasso Regression','Lasso');

CREATE TABLE model_top_features (
    id serial primary key,
    symbol character varying NOT NULL,
    model real NOT NULL,
    indicator character varying,
    rank int,
    unique (symbol, model)
);

ALTER TABLE public.symbols OWNER TO "mcohen";

--
-- Name: public; Type: ACL; Schema: -; Owner: Zipfian
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM "mcohen";
GRANT ALL ON SCHEMA public TO "mcohen";
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--
