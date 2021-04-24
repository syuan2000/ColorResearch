CREATE DATABASE `mauruemc_color`;

USE `mauruemc_color`;

DROP TABLE IF EXISTS `agents`;

CREATE TABLE  `Movies` 
   (	
    `id` int PRIMARY KEY AUTO_INCREMENT,
    `movieName` VARCHAR(50),
    `year` int,
    `director` VARCHAR(50),
    `cinematographer` VARCHAR(50),
    `aspectRatio` VARCHAR(50)
   );

DROP TABLE IF EXISTS `Frames`;

CREATE TABLE `Frames`
    (
    `frameID` bigint,
    `movieID` int references `Movies`(`id`),
    `filePath` VARCHAR(100),
    `avg_R` float,
    `avg_G` float,
    `avg_B` float,
    PRIMARY KEY(`frameID`, `movieID`)
    );

DROP TABLE IF EXISTS `Palettes`;

CREATE TABLE `Palettes`
    (
    `frame` bigint REFERENCES `Frames`(`frameID`),
    `colorRank` int,
    `top10_R` float,
    `top10_G` float, 
    `top10_B` float,
    `Kmeans_R` float,
    `Kmeans_G` float,
    `Kmeans_B` float, 
    `random10_R` float,
    `random10_G` float, 
    `random10_B` float,
    PRIMARY KEY(`frame`, `colorRank`)
    );